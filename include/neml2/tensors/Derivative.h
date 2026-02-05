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

#pragma once

#include "neml2/misc/types.h"
#include "neml2/tensors/Tensor.h"

namespace neml2
{
// forward declaration
template <std::size_t N>
class Derivative;

/// Pretty print derivative names
template <std::size_t N>
std::string
derivative_name(const std::string & var_name, const std::array<std::string, N> & arg_names)
{
  std::string name = "d";
  if constexpr (N > 1)
    name += std::to_string(N);
  name += "(" + var_name + ")/";
  for (const auto & arg_name : arg_names)
    name += "d(" + arg_name + ")";
  return name;
}

/**
 * Convert the derivative with broadcasting intrinsic intermediate dimensions into a Tensor
 * with full intrinsic intermediate dimensions. If this derivative is already full, simply
 * returns the underlying tensor.
 *
 * @tparam N The order of the derivative plus one
 * @param t The input derivative tensor
 * @param iid The number of intrinsic intermediate dimensions of the derivative
 * @param iiss The intrinsic intermediate shapes of the variable and arguments
 */
template <std::size_t N>
Tensor fullify(const Tensor & t, Size iid, std::array<TensorShape, N> iiss);

/// Derivative wrapper
template <std::size_t N>
class Derivative
{
public:
  Derivative() = default;
  Derivative(std::size_t intrsc_intmd_dim,
             const std::array<std::size_t, N + 1> & intrsc_intmd_dims,
             const std::array<TensorShapeRef, N + 1> & intmd_sizes,
             const std::array<TensorShapeRef, N + 1> & base_sizes,
             [[maybe_unused]] std::string var_name = "",
             [[maybe_unused]] std::array<std::string, N> arg_names = {});

  ///@{
  Derivative(const Derivative<N> & val) = default;
  Derivative(Derivative<N> && val) = default;
  ~Derivative() = default;
  ///@}

  /// Get the derivative name
  std::string name() const;
  /// Get the variable name
  const std::string & var_name() const;
  /// Get the i-th argument name
  const std::string & arg_name(std::size_t i) const;

  ///@{
  /// Assignment operator
  Derivative & operator=(const Tensor & val);
  Derivative & operator=(const Derivative<N> & val) = default;
  Derivative & operator=(Derivative<N> && val) noexcept = default;
  ///@}

  ///@{
  /// Compound assignment operator
  Derivative & operator+=(const Tensor & val);
  Derivative & operator+=(const Derivative<N> & val);
  ///@}

  /// Get the derivative
  const Tensor & tensor() const;

  /// Clear the derivative
  void clear();

  /// Whether this derivative contains defined value
  bool defined() const;

  /// Whether this derivative represents a broadcast over intrinsic intermediate dimensions
  bool is_intrsc_intmd_broadcast() const;

  ///@{
  /// Base shapes
  TensorShapeRef intmd_sizes() const;
  TensorShapeRef extrsc_intmd_sizes() const;
  TensorShapeRef intrsc_intmd_sizes() const;
  TensorShapeRef var_intmd_sizes() const;
  TensorShapeRef var_intrsc_intmd_sizes() const;
  TensorShapeRef arg_intmd_sizes(std::size_t i) const;
  TensorShapeRef arg_intrsc_intmd_sizes(std::size_t i) const;

  TensorShape base_sizes() const;
  TensorShapeRef var_base_sizes() const;
  TensorShapeRef arg_base_sizes(std::size_t i) const;
  ///@}

  /// Dimensions
  ///@{
  Size intmd_dim() const;
  Size intrsc_intmd_dim() const;
  Size var_intrsc_intmd_dim() const;
  Size arg_intrsc_intmd_dim(std::size_t i) const;

  Size base_dim() const;
  Size var_base_dim() const;
  Size arg_base_dim(std::size_t i) const;
  ///@}

  /// Reinterpret the derivative to add additional intrinsic intermediate dimensions to the variable and arguments
  Derivative<N> reinterpret(std::size_t additional_intrsc_intmd_dim) const;

  /**
   * Convert the derivative with broadcasting intrinsic intermediate dimensions into a Tensor
   * with full intrinsic intermediate dimensions. If this derivative is already full, simply
   * returns the underlying tensor.
   */
  Tensor fullify() const;

private:
  Tensor try_intmd_expand(const Tensor & val) const;

  /// Number of trailing intermediate dimensions of the derivative that are intrinsic to the model definition
  Size _intrsc_intmd_dim = 0;

  /// Intrinsic intermediate dimensions of the variable and arguments
  std::array<Size, N + 1> _intrsc_intmd_dims;

  /// Intermediate shapes of the variable and arguments
  std::array<TensorShape, N + 1> _intmd_sizes;

  /// Base shapes
  std::array<TensorShape, N + 1> _base_sizes;

  /// Debug name for error messages
  std::string _var_name;
  std::array<std::string, N> _arg_names;

  /// Derivative to write to
  Tensor _deriv;

  /// Cached intermediate dimension that this derivative last saw
  /// @note: set() and operator=() are the only methods that cache this. clear() does not invalidate the cache.
  Size _cached_intmd_dim = 0;
};
} // namespace neml2
