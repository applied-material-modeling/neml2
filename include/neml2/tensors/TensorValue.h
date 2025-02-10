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
enum class TensorType : int8_t;

/**
 * @brief The base class to allow us to set up a polymorphic container of Tensors. The concrete
 * definitions will be templated on the actual tensor type.
 *
 */
class TensorValueBase
{
public:
  TensorValueBase() = default;

  TensorValueBase(const TensorValueBase &) = default;
  TensorValueBase(TensorValueBase &&) noexcept = default;
  TensorValueBase & operator=(const TensorValueBase &) = default;
  TensorValueBase & operator=(TensorValueBase &&) noexcept = default;
  virtual ~TensorValueBase() = default;

  /// Send the value to the target options
  virtual void to_(const TensorOptions &) = 0;

  /// Require grad
  virtual void requires_grad_(bool req = true) = 0;

  /// Convert the parameter value to a Tensor
  virtual operator Tensor() const = 0;

  /// assignment operator
  virtual void operator=(const Tensor & val) = 0;

  /// Tensor type
  virtual TensorType type() const = 0;
};

/// Concrete definition of tensor value
template <typename T>
class TensorValue : public TensorValueBase
{
public:
  explicit TensorValue(T value)
    : _value(std::move(value))
  {
  }

  void to_(const TensorOptions & options) override;
  void requires_grad_(bool req = true) override;
  operator Tensor() const override;
  void operator=(const Tensor & val) override;
  TensorType type() const override;
  T & value() { return _value; }

private:
  T _value;
};
} // namespace neml2
