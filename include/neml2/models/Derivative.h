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

#include "neml2/tensors/Tensor.h"

namespace neml2
{
// Forward declaration
class VariableBase;

/// Derivative wrapper
template <std::size_t N>
class Derivative
{
public:
  Derivative() = default;
  Derivative(const std::array<const VariableBase *, N + 1> & var_and_args);

  /// Assignment operator
  ///@{
  Derivative & operator=(const Tensor & val);
  Derivative & operator=(const VariableBase & val);
  ///@}

  /// Equality operator (for searching in a container)
  bool operator==(const Derivative<N> & other) const;

  /// Get the derivative
  const Tensor & tensor() const { return _deriv; }

  /// Get the derivative in assembly format
  const Tensor & get() const;

  /// Set the derivative (given in assembly format)
  void set(const Tensor & val);

  /// Get the variable
  const VariableBase * var() const { return _var_and_args[0]; }

  /// Get the args
  std::array<const VariableBase *, N> args() const;

private:
  std::array<TensorShapeRef, N + 1> get_intmd_sizes() const;
  std::array<TensorShapeRef, N + 1> get_base_sizes() const;

  TensorShape total_intmd_sizes() const;
  TensorShape total_base_sizes() const;

  /// Variable and arguments (the first argument is the variable being differentiated)
  std::array<const VariableBase *, N + 1> _var_and_args;

  /// Derivative to write to
  Tensor _deriv;

  /// Derivative in assembly format
  mutable Tensor _deriv_assembly;

  /// Debug name for error messages
  const std::string _debug_name = "<anonymous>";
};
} // namespace neml2
