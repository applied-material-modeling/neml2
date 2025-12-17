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
///@{
/// Pretty print derivative names
std::string derivative_name(const std::string & var_name, const std::string & arg_name);
std::string derivative_name(const std::string & var_name,
                            const std::string & arg1_name,
                            const std::string & arg2_name);
///@}

/// Derivative wrapper
template <std::size_t N>
class Derivative
{
public:
  Derivative() = default;
  Derivative(const std::array<TensorShapeRef, N + 1> & intrsc_intmd_sizes,
             const std::array<TensorShapeRef, N + 1> & base_sizes,
             [[maybe_unused]] std::string debug_name = "<anonymous>",
             std::size_t intrsc_intmd_dim = 0);

  ///@{
  Derivative(const Derivative<N> & val) = default;
  Derivative(Derivative<N> && val) = default;
  ~Derivative() = default;
  ///@}

  /// Get the pretty-formatted name of this derivative
  const std::string & name() const { return _debug_name; }

  ///@{
  /// Assignment operator
  Derivative & operator=(const Tensor & val);
  Derivative & operator=(const Derivative<N> & val);
  Derivative & operator=(Derivative<N> && val) noexcept;
  ///@}

  ///@{
  /// Compound assignment operator
  Derivative & operator+=(const Tensor & val);
  Derivative & operator+=(const Derivative<N> & val);
  Derivative & operator+=(Derivative<N> && val) noexcept;
  ///@}

  /// Get the derivative
  const Tensor & tensor() const;

  /// Get the derivative in assembly format
  Tensor get() const;

  /// Set the derivative (given in assembly format)
  void set(const Tensor & val);

  /// Clear the derivative
  void clear();

  /// Whether this derivative contains defined value
  bool defined() const;

  /// Whether the derivative is broadcast along the intrinsic intermediate dimensions
  bool is_intrsc_intmd_broadcast() const;

  /// Shapes
  ///@{
  std::array<TensorShapeRef, N + 1> intrsc_intmd_sizes() const;
  TensorShapeRef intrsc_intmd_sizes(std::size_t i) const;
  TensorShape total_intrsc_intmd_sizes() const;

  std::array<TensorShapeRef, N + 1> base_sizes() const;
  TensorShapeRef base_sizes(std::size_t i) const;
  TensorShape total_base_sizes() const;
  ///@}

  /// Dimensions
  ///@{
  Size total_intmd_dim() const;

  std::array<Size, N + 1> intrsc_intmd_dim() const;
  Size intrsc_intmd_dim(std::size_t i) const;
  Size total_intrsc_intmd_dim() const;

  std::array<Size, N + 1> base_dim() const;
  Size base_dim(std::size_t i) const;
  Size total_base_dim() const;
  ///@}

private:
  Tensor try_intmd_expand(const Tensor & val) const;

  /// Dependent intermediate shapes
  const std::array<TensorShape, N + 1> _intrsc_intmd_sizes;

  /// Base shapes
  const std::array<TensorShape, N + 1> _base_sizes;

  /// Debug name for error messages
  const std::string _debug_name;

  /// Number of trailing intermediate dimensions of the derivative that are intrinsic to the model definition
  Size _intrsc_intmd_dim;

  /// Derivative to write to
  Tensor _deriv;

  /// Cached intermediate dimension that this derivative last saw
  /// @note: set() and operator=() are the only methods that cache this. clear() does not invalidate the cache.
  Size _cached_intmd_dim = 0;
};
} // namespace neml2
