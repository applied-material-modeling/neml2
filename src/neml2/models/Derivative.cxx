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
Derivative<N>::Derivative(const std::array<const VariableBase *, N + 1> & var_and_args)
  : _var_and_args(var_and_args)
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
  if (val.intmd_sizes() == intmd_sizes)
    assign_or_add(_deriv, val);
  else if (at::is_expandable_to(val.intmd_sizes(), var()->intmd_sizes()))
  {
    const auto broadcasted_intmd_sizes =
        N == 1
            ? utils::add_shapes(var()->intmd_sizes(), var()->intmd_sizes())
            : utils::add_shapes(var()->intmd_sizes(), var()->intmd_sizes(), var()->intmd_sizes());
    const auto aligned_intmd_sizes =
        N == 1 ? utils::add_shapes(
                     var()->intmd_sizes(),
                     utils::pad_prepend(_var_and_args[1]->intmd_sizes(), var()->intmd_dim()))
               : utils::add_shapes(
                     var()->intmd_sizes(),
                     utils::pad_prepend(_var_and_args[1]->intmd_sizes(), var()->intmd_dim()),
                     utils::pad_prepend(_var_and_args[2]->intmd_sizes(), var()->intmd_dim()));
    auto val2 = intmd_diagonalize(val.intmd_expand(var()->intmd_sizes()).intmd_flatten());
    if constexpr (N == 2)
      val2 = intmd_diagonalize(val2);
    val2 = intmd_sum_to_size(val2.intmd_reshape(broadcasted_intmd_sizes), aligned_intmd_sizes);

    assign_or_add(_deriv, val2.intmd_reshape(intmd_sizes));
  }
  // else if (at::is_expandable_to(val.intmd_sizes(), intmd_sizes))
  //   assign_or_add(_deriv, val.intmd_expand(intmd_sizes));
  else
    neml_assert_dbg(false,
                    "The assigned derivative for '",
                    _debug_name,
                    "' has incompatible intmd shape ",
                    val.intmd_sizes(),
                    ". Expected to be either expandable to the variable's intmd shape ",
                    var()->intmd_sizes(),
                    " or expandable to the derivative's intmd shape ",
                    intmd_sizes);

  // Invalidate the assembly cache
  _deriv_assembly = Tensor();

  return *this;
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
