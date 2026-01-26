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

#include "neml2/models/VariableBase.h"

namespace neml2
{
/**
 * @brief Concrete definition of a variable
 *
 */
template <typename T>
class Variable : public VariableBase
{
public:
  Variable(VariableName name_in, Model * owner, TensorShapeRef dep_intmd_dims = {})
    : VariableBase(std::move(name_in), owner, T::const_base_sizes, dep_intmd_dims),
      _ref(nullptr)
  {
  }

  TensorType type() const override;

  /// @name Tensor information
  // These methods mirror TensorBase
  ///@{
  /// Defined
  bool defined() const override;
  /// Tensor options
  TensorOptions options() const override;
  /// Scalar type
  Dtype scalar_type() const override;
  /// Device
  Device device() const override;
  ///@}

  const TraceableTensorShape & dynamic_sizes() const override;

  std::unique_ptr<VariableBase> clone(const VariableName & name = {},
                                      Model * owner = nullptr) const override;

  void ref(const VariableBase & var, bool ref_is_mutable = false) override;

  const VariableBase * ref() const override { return _ref ? _ref->ref() : this; }

  bool owning() const override { return !_ref; }

  void zero(const TensorOptions & options) override;

  void set(const Tensor & val, std::optional<TracerPrivilege> key) override;

  Tensor get() const override;

  Tensor tensor() const override;

  void requires_grad_(bool req = true) override;

  void operator=(const Tensor & val) override;

  /// Variable value
  const T & operator()() const { return owning() ? _value : (*_ref)(); }

  /// Negation
  T operator-() const { return -operator()(); }

  void clear() override;

protected:
  /// The variable referenced by this (nullptr if this is a storing variable)
  const Variable<T> * _ref;

  /// Whether mutating the referenced variable is allowed
  bool _ref_is_mutable = false;

  /// Variable value (undefined if this is a referencing variable)
  T _value;
};
} // namespace neml2
