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
#include "neml2/tensors/Derivative.h"
#include "neml2/misc/errors.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{

template <std::size_t N>
Derivative<N>::Derivative(std::size_t intrsc_intmd_dim,
                          const std::array<std::size_t, N + 1> & intrsc_intmd_dims,
                          const std::array<TensorShapeRef, N + 1> & intmd_sizes,
                          const std::array<TensorShapeRef, N + 1> & base_sizes,
                          [[maybe_unused]] std::string var_name,
                          [[maybe_unused]] std::array<std::string, N> arg_names)
  : _intrsc_intmd_dim(static_cast<Size>(intrsc_intmd_dim)),
    _intrsc_intmd_dims(
        std::apply([](auto... s) { return std::array{Size(s)...}; }, intrsc_intmd_dims)),
    _intmd_sizes(std::apply([](auto... s) { return std::array{TensorShape(s)...}; }, intmd_sizes)),
    _base_sizes(std::apply([](auto... s) { return std::array{TensorShape(s)...}; }, base_sizes)),
    _var_name(std::move(var_name)),
    _arg_names(std::move(arg_names))
{
#ifndef NDEBUG
  neml_assert_dbg(intrsc_intmd_dims[0] <= intmd_sizes[0].size(),
                  "The variable's intrinsic intermediate dimension (",
                  intrsc_intmd_dims[0],
                  ") of derivative '",
                  name(),
                  "' is larger than the variable's intermediate dimension (",
                  intmd_sizes[0].size(),
                  ").");
  for (std::size_t i = 0; i < N; ++i)
    neml_assert_dbg(intrsc_intmd_dims[i + 1] <= intmd_sizes[i + 1].size(),
                    "The argument ",
                    i,
                    "'s intrinsic intermediate dimension (",
                    intrsc_intmd_dims[i + 1],
                    ") of derivative '",
                    name(),
                    "' is larger than the argument ",
                    i,
                    "'s intermediate dimension (",
                    intmd_sizes[i + 1].size(),
                    ").");
  const auto total_intrsc_intmd_dim = utils::sum_array(intrsc_intmd_dims);
  neml_assert_dbg(
      intrsc_intmd_dim <= intrsc_intmd_dims[0] || intrsc_intmd_dim == total_intrsc_intmd_dim,
      "The intrinsic intermediate dimension (",
      intrsc_intmd_dim,
      ") of derivative '",
      name(),
      "' should be less than or equal to the variable's intrinsic intermediate dimension (",
      intrsc_intmd_dims[0],
      ") or equal to the sum of all intrinsic intermediate dimensions of the variable and its "
      "arguments (",
      total_intrsc_intmd_dim,
      ").");
#endif
}

template <std::size_t N>
std::string
Derivative<N>::name() const
{
  return derivative_name(_var_name, _arg_names);
}

template <std::size_t N>
const std::string &
Derivative<N>::var_name() const
{
  return _var_name;
}

template <std::size_t N>
const std::string &
Derivative<N>::arg_name(std::size_t i) const
{
  neml_assert_dbg(
      i < N, "Index out of bounds when accessing argument name of derivative '", name(), "'.");
  return _arg_names[i];
}

template <std::size_t N>
const Tensor &
Derivative<N>::tensor() const
{
  neml_assert_dbg(
      _deriv.defined() && _deriv.isfinite().all().template item<bool>(),
      "Derivative '",
      name(),
      "' has undefined or non-finite value. Below is some additional debugging information:\n",
      "  defined: ",
      _deriv.defined() ? "true" : "false",
      "\n  isfinite: ",
      _deriv.defined() && _deriv.isfinite().all().template item<bool>() ? "true" : "false");
  return _deriv;
}

template <std::size_t N>
void
Derivative<N>::clear()
{
  _deriv = Tensor();
}

template <std::size_t N>
bool
Derivative<N>::defined() const
{
  return _deriv.defined();
}

template <std::size_t N>
bool
Derivative<N>::is_intrsc_intmd_broadcast() const
{
  const auto total_intrsc_intmd_dim = utils::sum_array(_intrsc_intmd_dims);
  return _intrsc_intmd_dim < total_intrsc_intmd_dim;
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::intmd_sizes() const
{
  return _deriv.intmd_sizes();
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::extrsc_intmd_sizes() const
{
  neml_assert_dbg(_intrsc_intmd_dim <= _deriv.intmd_dim(),
                  "The intrinsic intermediate dimension (",
                  _intrsc_intmd_dim,
                  ") of derivative '",
                  name(),
                  "' is larger than the total intermediate dimension (",
                  _deriv.intmd_dim(),
                  ").");
  return _deriv.intmd_sizes().slice(0, _deriv.intmd_dim() - _intrsc_intmd_dim);
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::intrsc_intmd_sizes() const
{
  neml_assert_dbg(_intrsc_intmd_dim <= _deriv.intmd_dim(),
                  "The intrinsic intermediate dimension (",
                  _intrsc_intmd_dim,
                  ") of derivative '",
                  name(),
                  "' is larger than the total intermediate dimension (",
                  _deriv.intmd_dim(),
                  ").");
  return _deriv.intmd_sizes().slice(_deriv.intmd_dim() - _intrsc_intmd_dim);
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::var_intmd_sizes() const
{
  return _intmd_sizes[0];
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::var_intrsc_intmd_sizes() const
{
  const auto & s = _intmd_sizes[0];
  return TensorShapeRef(s).slice(s.size() - _intrsc_intmd_dims[0]);
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::arg_intmd_sizes(std::size_t i) const
{
  neml_assert_dbg(i < N,
                  "Index out of bounds when accessing argument intermediate shape of derivative '",
                  name(),
                  "'.");
  return _intmd_sizes[i + 1];
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::arg_intrsc_intmd_sizes(std::size_t i) const
{
  neml_assert_dbg(
      i < N,
      "Index out of bounds when accessing argument intrinsic intermediate shape of derivative '",
      name(),
      "'.");
  const auto & s = _intmd_sizes[i + 1];
  return TensorShapeRef(s).slice(s.size() - _intrsc_intmd_dims[i + 1]);
}

template <std::size_t N>
TensorShape
Derivative<N>::base_sizes() const
{
  return std::apply([](auto &&... s) { return utils::add_shapes(s...); }, _base_sizes);
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::var_base_sizes() const
{
  return _base_sizes[0];
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::arg_base_sizes(std::size_t i) const
{
  neml_assert_dbg(i < N,
                  "Index out of bounds when accessing argument base shape of derivative '",
                  name(),
                  "'.");
  return _base_sizes[i + 1];
}

template <std::size_t N>
Size
Derivative<N>::intmd_dim() const
{
  return _cached_intmd_dim;
}

template <std::size_t N>
Size
Derivative<N>::intrsc_intmd_dim() const
{
  return _intrsc_intmd_dim;
}

template <std::size_t N>
Size
Derivative<N>::var_intrsc_intmd_dim() const
{
  return _intrsc_intmd_dims[0];
}

template <std::size_t N>
Size
Derivative<N>::arg_intrsc_intmd_dim(std::size_t i) const
{
  neml_assert_dbg(i < N,
                  "Index out of bounds when accessing argument intrinsic intermediate dimension of "
                  "derivative '",
                  name(),
                  "'.");
  return _intrsc_intmd_dims[i + 1];
}

template <std::size_t N>
Size
Derivative<N>::base_dim() const
{
  return std::accumulate(_base_sizes.begin(),
                         _base_sizes.end(),
                         Size(0),
                         [](Size sum, const TensorShape & s) { return sum + s.size(); });
}

template <std::size_t N>
Size
Derivative<N>::var_base_dim() const
{
  return static_cast<Size>(_base_sizes[0].size());
}

template <std::size_t N>
Size
Derivative<N>::arg_base_dim(std::size_t i) const
{
  neml_assert_dbg(
      i < N, "Index out of bounds when accessing base dimension of derivative '", name(), "'.");
  return static_cast<Size>(_base_sizes[i + 1].size());
}

template <std::size_t N>
Derivative<N>
Derivative<N>::reinterpret(std::size_t) const
{
  throw NEMLException("Reinterpretation is not yet implemented for Derivative<" +
                      std::to_string(N) + ">.");
}

template <>
Derivative<1>
Derivative<1>::reinterpret(std::size_t additional_intrsc_intmd_dim) const
{
  neml_assert_dbg(defined(), "Cannot reinterpret an undefined derivative '", name(), "'.");
  neml_assert_dbg(
      intrsc_intmd_sizes() == var_intrsc_intmd_sizes(),
      "Derivative reinterpretation is currently only supported when derivative "
      "intrinsic intermediate shape matches variable intrinsic intermediate shape. Got ",
      intrsc_intmd_sizes(),
      " and ",
      var_intrsc_intmd_sizes(),
      ".");

  try
  {
    auto new_deriv = Derivative<1>(_intrsc_intmd_dim + additional_intrsc_intmd_dim,
                                   {_intrsc_intmd_dims[0] + additional_intrsc_intmd_dim,
                                    _intrsc_intmd_dims[1] + additional_intrsc_intmd_dim},
                                   {_intmd_sizes[0], _intmd_sizes[1]},
                                   {_base_sizes[0], _base_sizes[1]},
                                   _var_name,
                                   _arg_names);
    return new_deriv = _deriv;
  }
  catch (const NEMLException & e)
  {
    throw NEMLException(
        "Error during reinterpretating derivative '" + name() +
        "'. The original derivative had \n"
        "  intermediate shape " +
        utils::stringify(intmd_sizes()) + "\n  intrinsic intermediate dimension " +
        std::to_string(_intrsc_intmd_dim) + "\n  variable intrinsic intermediate dimension " +
        std::to_string(_intrsc_intmd_dims[0]) + "\n  argument intrinsic intermediate dimension " +
        std::to_string(_intrsc_intmd_dims[1]) + "\nThe reinterpretation attempted to add " +
        std::to_string(additional_intrsc_intmd_dim) +
        " intrinsic intermediate dimensions to both the variable and argument. Encountered "
        "error: \n" +
        e.what());
  }
}

template <std::size_t N>
Tensor
Derivative<N>::try_intmd_expand(const Tensor & deriv) const
{
  auto deriv2 = deriv;
  if (deriv2.intmd_dim() < _intrsc_intmd_dim)
    deriv2 = deriv2.intmd_unsqueeze(-1, _intrsc_intmd_dim - deriv2.intmd_dim());

  const auto deriv_intrsc_intmd_sizes =
      deriv2.intmd_sizes().slice(deriv2.intmd_dim() - _intrsc_intmd_dim);
  const auto total_intrsc_intmd_sizes =
      N == 1 ? utils::add_shapes(var_intrsc_intmd_sizes(), arg_intrsc_intmd_sizes(0))
             : utils::add_shapes(
                   var_intrsc_intmd_sizes(), arg_intrsc_intmd_sizes(0), arg_intrsc_intmd_sizes(1));

  // Easy case: intrinsic intermediate shape of the derivative matches the total intrinsice
  // intermediate shape -- no broadcasting needed
  if (deriv_intrsc_intmd_sizes == total_intrsc_intmd_sizes)
    return deriv2;

  // Broadcasting case: need to broadcast the intrinsic intermediate shape of the derivative
  // to the variable's intrinsic intermediate shape
  neml_assert_dbg(at::is_expandable_to(deriv_intrsc_intmd_sizes, var_intrsc_intmd_sizes()),
                  "The intrinsic intermediate shape (",
                  deriv_intrsc_intmd_sizes,
                  ") of the assigned derivative for '",
                  name(),
                  "' is not broadcastable to the variable's intrinsic intermediate shape (",
                  var_intrsc_intmd_sizes(),
                  ").");

  const auto extrsc_intmd_sizes =
      deriv2.intmd_sizes().slice(0, deriv2.intmd_dim() - _intrsc_intmd_dim);
  const auto target_intmd_sizes = utils::add_shapes(extrsc_intmd_sizes, var_intrsc_intmd_sizes());
  const auto deriv_aligned =
      deriv2.intmd_unsqueeze(-_intrsc_intmd_dim - 1, var_intrsc_intmd_dim() - _intrsc_intmd_dim);
  return deriv_aligned.intmd_expand(target_intmd_sizes);
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator=(const Tensor & val)
{
  neml_assert_dbg(val.base_sizes() == base_sizes(),
                  "The assigned derivative for '",
                  name(),
                  "' has incompatible base shape. Expected ",
                  base_sizes(),
                  ", got ",
                  val.base_sizes());

  _deriv = try_intmd_expand(val);
  _cached_intmd_dim = _deriv.intmd_dim();

  return *this;
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator+=(const Tensor & val)
{
  neml_assert_dbg(val.base_sizes() == base_sizes(),
                  "The assigned derivative for '",
                  name(),
                  "' has incompatible base shape. Expected ",
                  base_sizes(),
                  ", got ",
                  val.base_sizes());

  if (_deriv.defined())
    _deriv = _deriv + try_intmd_expand(val);
  else
    _deriv = try_intmd_expand(val);

  _cached_intmd_dim = _deriv.intmd_dim();

  return *this;
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator+=(const Derivative<N> & deriv)
{
  _intrsc_intmd_dim = deriv._intrsc_intmd_dim;
  return operator+=(deriv.tensor());
}

template class Derivative<1>;
template class Derivative<2>;
}
