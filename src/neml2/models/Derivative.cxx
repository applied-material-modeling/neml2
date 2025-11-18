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
#include "neml2/models/VariableBase.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/sum_to_size.h"
#include "neml2/tensors/functions/to_assembly.h"
#include "neml2/tensors/functions/from_assembly.h"

namespace neml2
{
template <std::size_t N>
Derivative<N>::Derivative(const std::array<const VariableBase *, N + 1> & var_and_args,
                          const std::array<ArrayRef<Size>, N + 1> & dep_dims)
  : _var_and_args(var_and_args),
    _dep_dims(normalize_dep_dims(dep_dims))
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
std::array<TensorShape, N + 1>
Derivative<N>::normalize_dep_dims(const std::array<ArrayRef<Size>, N + 1> & dep_dims) const
{
  std::array<TensorShape, N + 1> res{};
  for (std::size_t i = 0; i < N + 1; i++)
  {
    TensorShape dims;
    for (const auto & d : dep_dims[i])
      dims.push_back(utils::normalize_dim(d, 0, _var_and_args[i]->intmd_dim()));
    std::sort(dims.begin(), dims.end());
    res[i] = dims;
  }
  return res;
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

  // Otherwise, the intermediate shape must be broadcastable to the variable's intermediate
  // shape
  neml_assert_dbg(at::is_expandable_to(val.intmd_sizes(), var()->intmd_sizes()),
                  "The assigned derivative for '",
                  _debug_name,
                  "' has incompatible intermediate shape ",
                  val.intmd_sizes(),
                  ". Expected to be either expandable to the variable's intermediate shape ",
                  var()->intmd_sizes(),
                  " or match the derivative's full intermediate shape ",
                  intmd_sizes);

  // We'll go through a set of operations to get the right shape
  assign_or_add(_deriv, broadcast_intmd_dims(val));

  // Invalidate the assembly cache
  _deriv_assembly = Tensor();

  return *this;
}

template <std::size_t N>
Tensor
Derivative<N>::broadcast_intmd_dims(const Tensor & val) const
{
  auto val2 = val.intmd_expand(var()->intmd_sizes());

  // Move variable's dependent dims to the front
  const auto & var_dep_dims = _dep_dims[0];
  for (std::size_t i = 0; i < var_dep_dims.size(); i++)
    val2 = val2.intmd_movedim(var_dep_dims[i], Size(i));

  // Flatten independent dims
  const auto var_dep_sizes = val2.intmd_sizes().slice(0, var_dep_dims.size());
  const auto var_indep_sizes = val2.intmd_sizes().slice(var_dep_dims.size());
  const auto n = utils::numel(var_indep_sizes);
  val2 = val2.intmd_reshape(utils::add_shapes(var_dep_sizes, n));

  // Diagonalize independent dims
  for (std::size_t i = 0; i < N; i++)
    val2 = intmd_diagonalize(val2, -1);
  val2 = val2.intmd_reshape(utils::add_shapes(
      var_dep_sizes, var_indep_sizes, N == 2 ? var_indep_sizes : TensorShapeRef{}));

  // Move variable's dependent dims back to their original positions
  for (std::size_t i = var_dep_dims.size() - 1; i >= 0; i--)
    val2 = val2.intmd_movedim(Size(i), var_dep_dims[i]);

  // Unsqueeze and expand args' dependent dims
  auto d_offset = var()->intmd_dim();
  for (std::size_t i = 1; i <= N; i++)
  {
    const auto & arg_dep_dims = _dep_dims[i];
    for (const auto & d : arg_dep_dims)
      val2 = val2.intmd_unsqueeze(d_offset + d)
                 .intmd_expand(_var_and_args[i]->intmd_size(d), d_offset + d);
    d_offset += _var_and_args[i]->intmd_dim();
  }

  // Everything is in order now, let's sum_to_size
  return intmd_sum_to_size(val2, total_intmd_sizes());
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
