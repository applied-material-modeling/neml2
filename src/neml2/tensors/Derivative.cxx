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
#include "neml2/misc/types.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/to_assembly.h"
#include "neml2/tensors/functions/from_assembly.h"

namespace neml2
{
std::string
derivative_name(const std::string & var_name, const std::string & arg_name)
{
  return "d(" + var_name + ")/d(" + arg_name + ")";
}

std::string
derivative_name(const std::string & var_name,
                const std::string & arg1_name,
                const std::string & arg2_name)
{
  return "d2(" + var_name + ")/d(" + arg1_name + ")d(" + arg2_name + ")";
}

template <std::size_t N>
Derivative<N>::Derivative(const std::array<TensorShapeRef, N + 1> & intrsc_intmd_sizes,
                          const std::array<TensorShapeRef, N + 1> & base_sizes,
                          [[maybe_unused]] std::string debug_name,
                          std::size_t intrsc_intmd_dim)
  : _intrsc_intmd_sizes(
        std::apply([](auto... s) { return std::array{TensorShape(s)...}; }, intrsc_intmd_sizes)),
    _base_sizes(std::apply([](auto... s) { return std::array{TensorShape(s)...}; }, base_sizes)),
    _debug_name(std::move(debug_name)),
    _intrsc_intmd_dim(static_cast<Size>(intrsc_intmd_dim))
{
  neml_assert_dbg(
      _intrsc_intmd_dim <= this->intrsc_intmd_dim(0) ||
          _intrsc_intmd_dim == this->total_intrsc_intmd_dim(),
      "The intrinsic intermediate dimension (",
      _intrsc_intmd_dim,
      ") of derivative '",
      _debug_name,
      "' should be less than of equal to the variable's intrinsic intermediate dimension (",
      this->intrsc_intmd_dim(0),
      ") or equal to the total intrinsic intermediate dimension (",
      this->total_intrsc_intmd_dim(),
      ").");
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
  return _intrsc_intmd_dim <= total_intrsc_intmd_dim();
}

template <std::size_t N>
std::array<TensorShapeRef, N + 1>
Derivative<N>::intrsc_intmd_sizes() const
{
  std::array<TensorShapeRef, N + 1> refs;
  for (std::size_t i = 0; i <= N; ++i)
    refs[i] = _intrsc_intmd_sizes[i];
  return refs;
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::intrsc_intmd_sizes(std::size_t i) const
{
  neml_assert_dbg(i <= N,
                  "Index out of bounds when accessing intrinsic intermediate shape of derivative '",
                  _debug_name,
                  "'.");
  return _intrsc_intmd_sizes[i];
}

template <std::size_t N>
TensorShape
Derivative<N>::total_intrsc_intmd_sizes() const
{
  return std::apply([](auto &&... s) { return utils::add_shapes(s...); }, _intrsc_intmd_sizes);
}

template <std::size_t N>
std::array<TensorShapeRef, N + 1>
Derivative<N>::base_sizes() const
{
  std::array<TensorShapeRef, N + 1> refs;
  for (std::size_t i = 0; i <= N; ++i)
    refs[i] = _base_sizes[i];
  return refs;
}

template <std::size_t N>
TensorShapeRef
Derivative<N>::base_sizes(std::size_t i) const
{
  neml_assert_dbg(
      i <= N, "Index out of bounds when accessing base shape of derivative '", _debug_name, "'.");
  return _base_sizes[i];
}

template <std::size_t N>
TensorShape
Derivative<N>::total_base_sizes() const
{
  return std::apply([](auto &&... s) { return utils::add_shapes(s...); }, _base_sizes);
}

template <std::size_t N>
Size
Derivative<N>::total_intmd_dim() const
{
  return _cached_intmd_dim;
}

template <std::size_t N>
std::array<Size, N + 1>
Derivative<N>::intrsc_intmd_dim() const
{
  return std::apply([](auto &&... s)
                    { return std::array<Size, N + 1>{static_cast<Size>(s.size())...}; },
                    _intrsc_intmd_sizes);
}

template <std::size_t N>
Size
Derivative<N>::intrsc_intmd_dim(std::size_t i) const
{
  neml_assert_dbg(
      i <= N,
      "Index out of bounds when accessing intrinsic intermediate dimension of derivative '",
      _debug_name,
      "'.");
  return static_cast<Size>(_intrsc_intmd_sizes[i].size());
}

template <std::size_t N>
Size
Derivative<N>::total_intrsc_intmd_dim() const
{
  return std::accumulate(_intrsc_intmd_sizes.begin(),
                         _intrsc_intmd_sizes.end(),
                         Size(0),
                         [](Size sum, TensorShapeRef shape) { return sum + shape.size(); });
}

template <std::size_t N>
std::array<Size, N + 1>
Derivative<N>::base_dim() const
{
  return std::apply([](auto &&... s)
                    { return std::array<Size, N + 1>{static_cast<Size>(s.size())...}; },
                    _base_sizes);
}

template <std::size_t N>
Size
Derivative<N>::base_dim(std::size_t i) const
{
  neml_assert_dbg(i <= N,
                  "Index out of bounds when accessing base dimension of derivative '",
                  _debug_name,
                  "'.");
  return static_cast<Size>(_base_sizes[i].size());
}

template <std::size_t N>
Size
Derivative<N>::total_base_dim() const
{
  return std::accumulate(_base_sizes.begin(),
                         _base_sizes.end(),
                         Size(0),
                         [](Size sum, TensorShapeRef shape) { return sum + shape.size(); });
}

template <std::size_t N>
Tensor
Derivative<N>::try_intmd_expand(const Tensor & deriv) const
{
  neml_assert(deriv.intmd_dim() >= _intrsc_intmd_dim,
              "The intermediate dimension (",
              deriv.intmd_dim(),
              ") of the assigned derivative for '",
              _debug_name,
              "' is smaller than the declared intrinsic intermediate dimension (",
              _intrsc_intmd_dim,
              ").");

  if (_intrsc_intmd_dim == total_intrsc_intmd_dim())
  {
    neml_assert_dbg(deriv.intmd_sizes().slice(deriv.intmd_dim() - _intrsc_intmd_dim) ==
                        total_intrsc_intmd_sizes(),
                    "The intrinsic intermediate shape (",
                    deriv.intmd_sizes().slice(deriv.intmd_dim() - _intrsc_intmd_dim),
                    ") of the assigned derivative for '",
                    _debug_name,
                    "' does not match the declared intrinsic intermediate shape (",
                    total_intrsc_intmd_sizes(),
                    ").");
    return deriv;
  }

  const auto extrsc_intmd_sizes =
      deriv.intmd_sizes().slice(0, deriv.intmd_dim() - _intrsc_intmd_dim);
  const auto target_intmd_sizes = utils::add_shapes(extrsc_intmd_sizes, intrsc_intmd_sizes(0));
  const auto deriv_aligned =
      deriv.intmd_unsqueeze(-_intrsc_intmd_dim - 1, intrsc_intmd_dim(0) - _intrsc_intmd_dim);
  return deriv_aligned.intmd_expand(target_intmd_sizes);
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator=(const Derivative<N> & deriv)
{
  _intrsc_intmd_dim = deriv._intrsc_intmd_dim;
  return operator=(deriv.tensor());
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator=(Derivative<N> && deriv) noexcept
{
  _intrsc_intmd_dim = deriv._intrsc_intmd_dim;
  return operator=(std::move(deriv).tensor());
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator=(const Tensor & val)
{
  neml_assert_dbg(val.base_sizes() == total_base_sizes(),
                  "The assigned derivative for '",
                  _debug_name,
                  "' has incompatible base shape. Expected ",
                  total_base_sizes(),
                  ", got ",
                  val.base_sizes());

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

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator+=(Derivative<N> && deriv) noexcept
{
  _intrsc_intmd_dim = std::move(deriv)._intrsc_intmd_dim;
  return operator+=(std::move(deriv).tensor());
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator+=(const Tensor & val)
{
  neml_assert_dbg(val.base_sizes() == total_base_sizes(),
                  "The assigned derivative for '",
                  _debug_name,
                  "' has incompatible base shape. Expected ",
                  total_base_sizes(),
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
const Tensor &
Derivative<N>::tensor() const
{
  neml_assert_dbg(_deriv.defined(), "Derivative '", _debug_name, "' is not defined.");
  return _deriv;
}

template <std::size_t N>
Tensor
Derivative<N>::get() const
{
  neml_assert_dbg(_deriv.defined(), "Derivative '", _debug_name, "' is not defined.");
  return to_assembly<N + 1>(_deriv, intrsc_intmd_sizes(), base_sizes(), _debug_name);
}

template <std::size_t N>
void
Derivative<N>::set(const Tensor & val)
{
  neml_assert_dbg(
      val.defined(), "Derivative '", _debug_name, "' is assigned with undefined value.");
  _deriv = from_assembly<N + 1>(val, intrsc_intmd_sizes(), base_sizes(), _debug_name);
  _cached_intmd_dim = _deriv.intmd_dim();
}

template class Derivative<1>;
template class Derivative<2>;
}
