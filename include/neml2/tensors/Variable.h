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

#include "neml2/tensors/LabeledAxisAccessor.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"
#include "neml2/base/DependencyResolver.h"

namespace neml2
{
using VariableName = LabeledAxisAccessor;

// Forward declarations
class Model;
class Derivative;

class VariableBase
{
public:
  VariableBase(const VariableName & name_in,
               const Model * owner,
               FType ftype,
               TensorShapeRef list_shape);

  virtual ~VariableBase() = default;

  /// Name of this variable
  const VariableName & name() const { return _name; }

  /// The Model who declared this variable
  const Model & owner() const { return *_owner; }

  /// Variable ftype (role in a function definition)
  FType ftype() const { return _ftype; }

  /// Variable tensor type
  virtual TensorType type() const = 0;

  /// @name Subaxis
  ///@{
  bool is_state() const;
  bool is_old_state() const;
  bool is_force() const;
  bool is_old_force() const;
  bool is_residual() const;
  bool is_parameter() const;
  bool is_solve_dependent() const;
  /// Check if the derivative with respect to this variable should be evaluated
  // Note that the check depends on whether we are currently solving nonlinear system
  bool is_dependent() const;
  ///@}

  /// @name Tensor information
  // These methods mirror TensorBase
  ///@{
  /// Tensor options
  torch::TensorOptions options() const { return get().options(); }
  /// Scalar type
  torch::Dtype scalar_type() const { return get().scalar_type(); }
  /// Device
  torch::Device device() const { return get().device(); }
  /// Number of tensor dimensions
  Size dim() const { return get().dim(); }
  /// Tensor shape
  TensorShapeRef sizes() const { return get().sizes(); }
  /// Size of a dimension
  Size size(Size dim) const { return get().size(dim); }
  /// Whether the tensor is batched
  bool batched() const { return get().batched(); }
  /// Return the number of batch dimensions
  Size batch_dim() const { return get().batch_dim(); }
  /// Return the number of list dimensions
  Size list_dim() const { return list_sizes().size(); }
  /// Return the number of base dimensions
  Size base_dim() const { return base_sizes().size(); }
  /// Return the batch shape
  const TraceableTensorShape & batch_sizes() const { return get().batch_sizes(); }
  /// Return the list shape
  TensorShapeRef list_sizes() const { return _list_sizes; }
  /// Return the size of a batch axis
  TraceableSize batch_size(Size dim) const { return get().batch_size(dim); }
  /// Return the base shape
  virtual TensorShapeRef base_sizes() const = 0;
  /// Return the size of a base axis
  Size base_size(Size dim) const { return base_sizes()[dim]; }
  /// Return the size of a list axis
  Size list_size(Size dim) const { return list_sizes()[dim]; }
  /// Base storage of the variable
  Size base_storage() const { return utils::storage_size(base_sizes()); }
  /// Assembly storage of the variable
  Size assembly_storage() const
  {
    return utils::storage_size(list_sizes()) * utils::storage_size(base_sizes());
  }
  ///@}

  /// Clone this variable
  virtual std::unique_ptr<VariableBase> clone(const VariableName & name = {},
                                              const Model * owner = nullptr,
                                              FType ftype = FType::NONE) const = 0;

  /// Reference another variable
  virtual void ref(const VariableBase & other, bool ref_is_mutable = false) = 0;

  /// Get the referencing variable (returns this if this is a storing variable)
  virtual const VariableBase * ref() const = 0;

  /// Set the variable value
  virtual void set(const Tensor & val) = 0;

  /// Get the variable value
  virtual Tensor get() const = 0;

  /// Assignment operator
  virtual void operator=(const Tensor & val) = 0;

  /// Wrapper for assigning partial derivative
  Derivative d(const VariableBase & var);

  /// Wrapper for assigning second partial derivative
  Derivative d(const VariableBase & var1, const VariableBase & var2);

  /// Partial derivatives
  const std::map<VariableName, Tensor> & derivatives() const { return _derivs; }
  std::map<VariableName, Tensor> & derivatives() { return _derivs; }

  /// Partial second derivatives
  const std::map<VariableName, std::map<VariableName, Tensor>> & second_derivatives() const
  {
    return _sec_derivs;
  }
  std::map<VariableName, std::map<VariableName, Tensor>> & second_derivatives()
  {
    return _sec_derivs;
  }

  /// Clear the variable value and derivatives
  virtual void clear();

  /// Apply first order chain rule
  void apply_chain_rule(const DependencyResolver<Model, VariableName> &);

  /// Apply second order chain rule
  void apply_second_order_chain_rule(const DependencyResolver<Model, VariableName> &);

protected:
  /// Name of the variable
  const VariableName _name;

  /// The model which declared this variable
  const Model * const _owner;

  /// Variable ftype (role in a function definition)
  const FType _ftype;

  /// List shape of the variable
  const TensorShape _list_sizes;

  /// Derivatives of this variable with respect to other variables
  std::map<VariableName, Tensor> _derivs;

  /// Second derivatives of this variable with respect to other variables
  std::map<VariableName, std::map<VariableName, Tensor>> _sec_derivs;

private:
  std::map<VariableName, Tensor>
  total_derivatives(const DependencyResolver<Model, VariableName> & dep,
                    Model * model,
                    const VariableName & yvar) const;

  std::map<VariableName, std::map<VariableName, Tensor>>
  total_second_derivatives(const DependencyResolver<Model, VariableName> & dep,
                           Model * model,
                           const VariableName & yvar) const;
};

/**
 * @brief Concrete definition of a variable
 *
 */
template <typename T>
class Variable : public VariableBase
{
public:
  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<Tensor, T2>>>
  Variable(const VariableName & name_in,
           const Model * owner,
           FType ftype,
           TensorShapeRef list_shape)
    : VariableBase(name_in, owner, ftype, list_shape),
      _base_sizes(T::const_base_sizes),
      _ref(nullptr),
      _ref_is_mutable(false)
  {
  }

  template <typename T2 = T, typename = typename std::enable_if_t<std::is_same_v<Tensor, T2>>>
  Variable(const VariableName & name_in,
           const Model * owner,
           FType ftype,
           TensorShapeRef list_shape,
           TensorShapeRef base_shape)
    : VariableBase(name_in, owner, ftype, list_shape),
      _base_sizes(base_shape),
      _ref(nullptr),
      _ref_is_mutable(false)
  {
  }

  TensorType type() const override { return TensorTypeEnum<T>::value; }

  TensorShapeRef base_sizes() const override { return _base_sizes; }

  std::unique_ptr<VariableBase> clone(const VariableName & name = {},
                                      const Model * owner = nullptr,
                                      FType ftype = FType::NONE) const override
  {
    if constexpr (std::is_same_v<T, Tensor>)
    {
      return std::make_unique<Variable<Tensor>>(name.empty() ? this->name() : name,
                                                owner ? owner : &this->owner(),
                                                ftype != FType::NONE ? ftype : this->ftype(),
                                                list_sizes(),
                                                base_sizes());
    }
    else
    {
      return std::make_unique<Variable<T>>(name.empty() ? this->name() : name,
                                           owner ? owner : &this->owner(),
                                           ftype != FType::NONE ? ftype : this->ftype(),
                                           list_sizes());
    }
  }

  void ref(const VariableBase & var, bool ref_is_mutable = false) override
  {
    const auto * var_ptr = dynamic_cast<const Variable<T> *>(var.ref());
    neml_assert(var_ptr,
                "Variable ",
                name(),
                " of type ",
                type(),
                " failed to reference another variable named ",
                var.name(),
                " of type ",
                var.type(),
                ": Dynamic cast failure.");
    _ref = var_ptr;
    _ref_is_mutable = ref_is_mutable;
  }

  const VariableBase * ref() const override { return _ref ? _ref->ref() : this; }

  void set(const Tensor & val) override
  {
    if (_ref)
    {
      neml_assert_dbg(_ref_is_mutable,
                      "Trying to assign value to a referencing variable, but the referenced "
                      "variable is not mutable.");
      const_cast<Variable<T> *>(_ref)->set(val);
    }
    else
    {
      _value = T(val.base_reshape(utils::add_shapes(list_sizes(), base_sizes())),
                 utils::add_traceable_shapes(val.batch_sizes(), list_sizes()));
    }
  }

  Tensor get() const override
  {
    if (_ref)
      return _ref->get();

    auto batch_sizes = _value.batch_sizes().slice(0, _value.batch_dim() - list_dim());
    return Tensor(_value, batch_sizes);
  }

  /// Suppressed constructor to prevent accidental dereferencing
  [[deprecated("Variable<T> must be assigned to references -- missing &")]] Variable(
      const Variable<T> &&)
  {
  }

  /// Suppressed constructor to prevent accidental dereferencing
  [[deprecated("Variable<T> must be assigned to references -- missing &")]] Variable(
      const Variable<T> &)
  {
  }

  /// Suppressed assignment operator to prevent accidental dereferencing
  [[deprecated("Variable<T> must be assigned to references -- missing &")]] void
  operator=(const Variable<T> &)
  {
  }

  /// Set the variable value
  void operator=(const Tensor & val) override
  {
    if (_ref)
    {
      neml_assert_dbg(_ref_is_mutable,
                      "Trying to assign value to a referencing variable, but the referenced "
                      "variable is not mutable.");
      *const_cast<Variable<T> *>(_ref) = val;
    }
    else
      _value = T(val);
  }

  /// Variable value
  const T & value() const { return _ref ? _ref->value() : _value; }

  /// Negation
  T operator-() const { return -value(); }

  operator T() const { return value(); }

  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<T2, Tensor>>>
  operator Tensor() const
  {
    return value();
  }

  void clear() override
  {
    VariableBase::clear();

    if (!_ref)
      _value = T();
  }

protected:
  /// Base shape of the variable
  const TensorShape _base_sizes;

  /// The variable referenced by this (nullptr if this is a storing variable)
  const Variable<T> * _ref;

  /// Whether mutating the referenced variable is allowed
  bool _ref_is_mutable;

  /// Variable value (undefined if this is a referencing variable)
  T _value;
};

class Derivative
{
public:
  Derivative()
    : _base_sizes({}),
      _deriv(nullptr)
  {
  }

  Derivative(TensorShapeRef base_sizes, Tensor * deriv)
    : _base_sizes(base_sizes),
      _deriv(deriv)
  {
  }

  void operator=(const Tensor & val);

private:
  /// Base shape of the derivative
  const TensorShape _base_sizes;

  /// Derivative to write to
  Tensor * const _deriv;
};

// Everything below is just for convenience: We just forward operations to the the variable values
// so that we can do
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//   var4 = (var1 - var2) * var3
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// instead of the (ugly?) expression below
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//   var4 = (var1.v - var2.v) * var3.v
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define FWD_VARIABLE_BINARY_OP(op)                                                                 \
  template <typename T1,                                                                           \
            typename T2,                                                                           \
            typename = typename std::enable_if_t<std::is_base_of_v<VariableBase, T1> ||            \
                                                 std::is_base_of_v<VariableBase, T2>>>             \
  auto op(const T1 & a, const T2 & b)                                                              \
  {                                                                                                \
    if constexpr (std::is_base_of_v<VariableBase, T1> && std::is_base_of_v<VariableBase, T2>)      \
      return op(a.value(), b.value());                                                             \
                                                                                                   \
    if constexpr (std::is_base_of_v<VariableBase, T1> && !std::is_base_of_v<VariableBase, T2>)     \
      return op(a.value(), b);                                                                     \
                                                                                                   \
    if constexpr (!std::is_base_of_v<VariableBase, T1> && std::is_base_of_v<VariableBase, T2>)     \
      return op(a, b.value());                                                                     \
  }                                                                                                \
  static_assert(true)
FWD_VARIABLE_BINARY_OP(operator+);
FWD_VARIABLE_BINARY_OP(operator-);
FWD_VARIABLE_BINARY_OP(operator*);
FWD_VARIABLE_BINARY_OP(operator/);
}
