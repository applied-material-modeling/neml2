// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <ATen/ExpandUtils.h>

#include "neml2/misc/assertions.h"
#include "neml2/models/Derivative.h"
#include "neml2/misc/types.h"
#include "neml2/models/VariableBase.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/sum_to_size.h"
#include "neml2/tensors/functions/to_assembly.h"
#include "neml2/tensors/functions/from_assembly.h"

namespace neml2
{
static TensorShape
normalize_dims(ArrayRef<Size> dims, Size lb, Size ub)
{
  TensorShape res(dims.size());
  for (std::size_t i = 0; i < dims.size(); i++)
    res[i] = utils::normalize_dim(dims[i], lb, ub);
  std::sort(res.begin(), res.end());
  return res;
}

static TensorShape
remove_dims(TensorShapeRef shape, const TensorShape & dims_to_remove)
{
  TensorShape res{shape};
  for (std::size_t i = 0; i < dims_to_remove.size(); i++)
    res.erase(res.begin() + dims_to_remove[i] - Size(i));
  return res;
}

template <std::size_t N>
Derivative<N>::Derivative(const std::array<const VariableBase *, N + 1> & var_and_args,
                          ArrayRef<Size> dep_dims)
  : _var_and_args(var_and_args),
    _dep_dims(dep_dims)
#ifndef NDEBUG
    ,
    _debug_name(N == 1 ? std::string("d(") + var_and_args[0]->name().str() + ")/d(" +
                             var_and_args[1]->name().str() + ")"
                       : std::string("d2(") + var_and_args[0]->name().str() + ")/d(" +
                             var_and_args[1]->name().str() + ")d(" + var_and_args[2]->name().str() +
                             ")")
#endif
{
}

template <std::size_t N>
std::array<const VariableBase *, N>
Derivative<N>::args() const
{
  std::array<const VariableBase *, N> a{};
  std::copy(_var_and_args.begin() + 1, _var_and_args.end(), a.begin());
  return a;
}

template <std::size_t N>
std::array<TensorShapeRef, N + 1>
Derivative<N>::get_intmd_sizes() const
{
  return std::apply([&](auto &&... i) { return std::array{i->intmd_sizes()...}; }, _var_and_args);
}

template <std::size_t N>
std::array<TensorShapeRef, N + 1>
Derivative<N>::get_base_sizes() const
{
  return std::apply([&](auto &&... i) { return std::array{i->base_sizes()...}; }, _var_and_args);
}

template <std::size_t N>
TensorShape
Derivative<N>::total_intmd_sizes() const
{
  return std::apply([&](auto &&... args) { return utils::add_shapes(args...); }, get_intmd_sizes());
}

template <std::size_t N>
TensorShape
Derivative<N>::total_base_sizes() const
{
  return std::apply([&](auto &&... args) { return utils::add_shapes(args...); }, get_base_sizes());
}

static void
assign_or_add(Tensor & dest, const Tensor & val)
{
  if (dest.defined())
    dest = dest + val;
  else
    dest = val;
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator=(const Tensor & val)
{
#ifndef NDEBUG
  const auto base_sizes = total_base_sizes();
  neml_assert_dbg(val.base_sizes() == base_sizes,
                  "The assigned derivative for '",
                  _debug_name,
                  "' has incompatible base shape. Expected ",
                  base_sizes,
                  ", got ",
                  val.base_sizes());
#endif

  const auto intmd_sizes = total_intmd_sizes();

  // Easiest case if we have the full intermediate shape
  if (val.intmd_sizes() == intmd_sizes)
    assign_or_add(_deriv, val);
  else
  {
    // We'll go through a set of operations to get the right shape
    assign_or_add(_deriv, broadcast_intmd_dims(val));
  }

  // Invalidate the assembly cache
  _deriv_assembly = Tensor();

  return *this;
}

template <std::size_t N>
Tensor
Derivative<N>::broadcast_intmd_dims(const Tensor & val) const
{
  // Using this concrete example to illustrate the procedure:
  //   var:  (...; 7, 2, _2_, 3; ...)
  //   arg1: (...; _4_, 1, 3; ...)
  //   arg2: (...; 2, 1, _9_; ...)
  // Underscored dimensions are _dependent_ dimensions tagged in the variable/arguments
  //
  // Suppose we are given a derivative tensor 'val' with shape
  //   val:  (...; 2, _2_, 1, _4_, _9_; ...)
  //
  // The derivative must have the dependent dimensions tagged as well, in the order of
  // var, arg1, arg2.

  // Normalize and sort derivative's dependent dims
  // After this, dep_dims should be (1, 3, 4) in the example above
  const auto dep_dims = normalize_dims(_dep_dims, 0, val.intmd_dim());

  // Normalize and sort variable and args' dependent dims
  // After this, var_and_args_dep_dims should be {2}, {0}, {2} in the example above
  std::array<TensorShape, N + 1> var_and_args_dep_dims{};
  std::transform(_var_and_args.begin(),
                 _var_and_args.end(),
                 var_and_args_dep_dims.begin(),
                 [](const auto * x)
                 { return normalize_dims(x->dep_intmd_dims(), 0, x->intmd_dim()); });

  // Each dependent dimension must have the same size as the corresponding tagged dimension
  neml_assert_dbg(dep_dims.size() == std::accumulate(var_and_args_dep_dims.begin(),
                                                     var_and_args_dep_dims.end(),
                                                     std::size_t{0},
                                                     [](std::size_t a, const auto & s)
                                                     { return a + s.size(); }),
                  "The number of dependent dimensions in the assigned derivative for '",
                  _debug_name,
                  "' does not match the total number of tagged dependent dimensions in the "
                  "variable and its arguments.");

#ifndef NDEBUG
  std::size_t d = 0;
  for (std::size_t i = 0; i < N + 1; i++)
    for (const auto & dd : var_and_args_dep_dims[i])
    {
      neml_assert_dbg(val.intmd_size(dep_dims[d]) == _var_and_args[i]->intmd_size(dd),
                      "The size of dependent dimension ",
                      dep_dims[d],
                      " in the assigned derivative for '",
                      _debug_name,
                      "' does not match the size of tagged dependent dimension ",
                      dd,
                      " of size ",
                      _var_and_args[i]->intmd_size(dd),
                      " in variable '",
                      _var_and_args[i]->name().str(),
                      "'.");
      d++;
    }
#endif

  // Move dependent dims to the end
  // After this, val2 should have shape (...; 2, 1, _2_, _4_, _9_; ...)
  auto val2 = val;
  for (std::size_t i = 0; i < dep_dims.size(); i++)
    val2 = val2.intmd_movedim(dep_dims[i] - Size(i), -1);

  // Get the dependent shape
  // After this, using the example above
  //   indep_sizes should be (2, 1)
  //   dep_sizes should be (2, 4, 9)
  const auto indep_sizes = val2.intmd_sizes().slice(0, val2.intmd_dim() - dep_dims.size());
  const auto dep_sizes = val2.intmd_sizes().slice(val2.intmd_dim() - dep_dims.size());

  // The derivative's independent shape must be broadcastable to the variable's independent shape
  // Here, var_indep_sizes should be (7, 2, 3)
  const auto var_indep_sizes = remove_dims(var()->intmd_sizes(), var_and_args_dep_dims[0]);
  neml_assert_dbg(at::is_expandable_to(indep_sizes, var_indep_sizes),
                  "The independent intermediate shape ",
                  indep_sizes,
                  " of the assigned derivative for '",
                  _debug_name,
                  "' is not broadcastable to the variable's independent intermediate shape ",
                  var_indep_sizes,
                  ".");

  // Expand and flatten the derivative's independent dims
  // After this, val2 should have shape (...; 42, _2_, _4_, _9_; ...)
  val2 = val2.intmd_expand(utils::add_shapes(var_indep_sizes, dep_sizes))
             .intmd_reshape(utils::add_shapes(utils::numel(var_indep_sizes), dep_sizes));

  // Diagonalize the flattened independent dim for each argument
  // After this, val2 should have shape (...; 42, 42, 42, _2_, _4_, _9_; ...)
  for (std::size_t i = 1; i <= N; i++)
    val2 = intmd_diagonalize(val2, 0);

  // Unflatten the independent dims
  // After this, val2 should have shape (...; 7, 2, 3, 7, 2, 3, 7, 2, 3, _2_, _4_, _9_; ...)
  val2 = val2.intmd_reshape(utils::add_shapes(
      var_indep_sizes, var_indep_sizes, N == 2 ? var_indep_sizes : TensorShapeRef{}, dep_sizes));

  // Some argument's independent dimensions might have been broadcasted, so we need to perform a
  // reduction along those dims.
  // Here,
  //   total_indep_sizes should be (7, 2, 3, 3, 2, 1)
  //   aligned_indep_sizes should be (7, 2, 3, 1, 1, 3, 1, 2, 1)
  // note the padded ones in aligned_indep_sizes for alignment.
  // After this, val2 should have shape (...; 7, 2, 3, 3, 2, 1, _2_, _4_, _9_; ...)
  const auto arg1_indep_sizes =
      remove_dims(_var_and_args[1]->intmd_sizes(), var_and_args_dep_dims[1]);
  neml_assert_dbg(at::is_expandable_to(arg1_indep_sizes, var_indep_sizes),
                  "The independent intermediate shape ",
                  arg1_indep_sizes,
                  " of argument variable '",
                  _var_and_args[1]->name().str(),
                  "' is not broadcastable to the variable's independent intermediate shape ",
                  var_indep_sizes,
                  ".");
  auto total_indep_sizes = utils::add_shapes(var_indep_sizes, arg1_indep_sizes);
  auto aligned_indep_sizes = utils::add_shapes(
      var_indep_sizes, utils::pad_prepend(arg1_indep_sizes, var_indep_sizes.size()));
  if constexpr (N == 2)
  {
    const auto arg2_indep_sizes =
        remove_dims(_var_and_args[2]->intmd_sizes(), var_and_args_dep_dims[2]);
    neml_assert_dbg(at::is_expandable_to(arg2_indep_sizes, var_indep_sizes),
                    "The independent intermediate shape ",
                    arg2_indep_sizes,
                    " of argument variable '",
                    _var_and_args[2]->name().str(),
                    "' is not broadcastable to the variable's independent intermediate shape ",
                    var_indep_sizes,
                    ".");
    total_indep_sizes = utils::add_shapes(total_indep_sizes, arg2_indep_sizes);
    aligned_indep_sizes = utils::add_shapes(
        aligned_indep_sizes, utils::pad_prepend(arg2_indep_sizes, var_indep_sizes.size()));
  }
  val2 = intmd_sum_to_size(val2, utils::add_shapes(aligned_indep_sizes, dep_sizes))
             .intmd_reshape(utils::add_shapes(total_indep_sizes, dep_sizes));

  // Finally, move the dependent dims back to their original positions
  // Before this, val2 havs shape (...; 7, 2, 3, 3, 2, 1, _2_, _4_, _9_; ...)
  // After this, val2 should have shape (...; 7, 2, _2_, 3, _4_, 1, 3; 2, 1, _9_; ...)
  auto d0 = -Size(dep_dims.size());
  Size d_offset = 0;
  for (std::size_t i = 0; i < N + 1; i++)
  {
    for (const auto & dd : var_and_args_dep_dims[i])
    {
      const auto d1 = d_offset + dd;
      val2 = val2.intmd_movedim(d0, d1);
      d0++;
    }
    d_offset += _var_and_args[i]->intmd_dim();
  }

  neml_assert_dbg(val2.intmd_sizes() == total_intmd_sizes(),
                  "Internal error in broadcasting dependent dimensions for derivative '",
                  _debug_name,
                  "'.");
  return val2;
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator=(const VariableBase & val)
{
  return Derivative<N>::operator=(val.tensor());
}

template <std::size_t N>
bool
Derivative<N>::operator==(const Derivative<N> & other) const
{
  for (std::size_t i = 0; i < N + 1; i++)
    if (_var_and_args[i] != other._var_and_args[i])
      return false;

  return true;
}

template <std::size_t N>
const Tensor &
Derivative<N>::get() const
{
  if (_deriv_assembly.defined())
    return _deriv_assembly;

  if (_deriv.defined())
    _deriv_assembly = to_assembly<N + 1>(_deriv, get_intmd_sizes(), get_base_sizes(), _debug_name);

  return _deriv_assembly;
}

template <std::size_t N>
void
Derivative<N>::set(const Tensor & val)
{
  _deriv_assembly = val;
  _deriv = from_assembly<N + 1>(val, get_intmd_sizes(), get_base_sizes(), _debug_name);
}

template class Derivative<1>;
template class Derivative<2>;

}
