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

#include "neml2/models/map_types_fwd.h"
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Tensor.h"

namespace neml2
{
// Forward declarations
class Model;
template <std::size_t N>
class Derivative;
enum class TensorType : int8_t;
template <typename, typename>
class DependencyResolver;
struct TraceableSize;
struct TraceableTensorShape;
class VariableStore;

/**
 * @brief Base class of variable
 *
 * Specific implementations are defined by the derived class `Variable<T>` where we rely on
 * polymorphism so that we can store different types of variables in the same container.
 *
 */
class VariableBase
{
public:
  VariableBase() = default;

  VariableBase(const VariableBase &) = delete;
  VariableBase(VariableBase &&) = delete;
  VariableBase & operator=(const VariableBase &) = delete;
  VariableBase & operator=(VariableBase &&) = delete;
  virtual ~VariableBase() = default;

  VariableBase(VariableName name_in, Model * owner, TensorShapeRef lbatch_shape);

  /// Name of this variable
  const VariableName & name() const { return _name; }

  ///@{
  /// The Model who declared this variable
  const Model & owner() const;
  Model & owner();
  ///@}

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
  /// Defined
  bool defined() const;
  /// Tensor options
  TensorOptions options() const;
  /// Scalar type
  Dtype scalar_type() const;
  /// Device
  Device device() const;
  /// Number of tensor dimensions
  Size dim() const;
  /// Tensor shape
  TensorShapeRef sizes() const;
  /// Size of a dimension
  Size size(Size dim) const;
  /// Whether the tensor is batched
  bool batched() const;
  /// Return the number of left-batch dimensions
  Size lbatch_dim() const;
  /// Return the number of batch dimensions
  Size batch_dim() const;
  /// Return the number of base dimensions
  Size base_dim() const;
  /// Return the left-batch shape
  TensorShapeRef lbatch_sizes() const;
  /// Return the batch shape
  TraceableTensorShape batch_sizes() const;
  /// Return the base shape
  virtual TensorShapeRef base_sizes() const = 0;
  /// Return the size of a left-batch axis
  Size lbatch_size(Size dim) const;
  /// Return the size of a batch axis
  TraceableSize batch_size(Size dim) const;
  /// Return the size of a base axis
  Size base_size(Size dim) const;
  /// Base storage of the variable
  Size base_storage() const;
  /// Assembly storage of the variable
  Size assembly_storage() const;
  ///@}

  /// Clone this variable
  virtual std::unique_ptr<VariableBase> clone(const VariableName & name = {},
                                              Model * owner = nullptr) const = 0;

  /// Reference another variable
  virtual void ref(const VariableBase & other, bool ref_is_mutable = false) = 0;

  /// Get the referencing variable (returns this if this is a storing variable)
  virtual const VariableBase * ref() const = 0;

  /// Check if this is an owning variable
  virtual bool owning() const = 0;

  /// Make a zeros tensor
  Tensor make_zeros(const TraceableTensorShape & batch_shape, const TensorOptions & options) const;

  /// Make a zeros derivative
  Tensor make_zeros(const VariableBase & xvar,
                    const TraceableTensorShape & batch_shape,
                    const TensorOptions & options) const;

  /// Make a zeros second derivative
  Tensor make_zeros(const VariableBase & x1var,
                    const VariableBase & x2var,
                    const TraceableTensorShape & batch_shape,
                    const TensorOptions & options) const;

  /// Set the variable value to zero
  virtual void zero(const TensorOptions & options) = 0;

  /// Set the variable value from a Tensor in assembly format
  virtual void set(const Tensor & val) = 0;

  /// Get the variable value in assembly format
  virtual Tensor get() const = 0;

  /// Get the variable value cast to Tensor
  virtual Tensor tensor() const = 0;

  /// Check if this variable is part of the AD function graph
  bool requires_grad() const;

  /// Mark this variable as a leaf variable in tracing function graph for AD
  virtual void requires_grad_(bool req = true) = 0;

  /// Assignment operator
  virtual void operator=(const Tensor & val) = 0;

  /// Secret passkey to control access to the secret assignment method
  struct RawAssignment
  {
    RawAssignment(const RawAssignment &) = default;
    RawAssignment(RawAssignment &&) = default;
    RawAssignment & operator=(const RawAssignment &) = delete;
    RawAssignment & operator=(RawAssignment &&) = delete;
    ~RawAssignment() = default;

  private:
    // only the following classes can construct this key
    friend class VariableStore;

    // private constructor so only trusted friends can construct this key
    RawAssignment() = default;
  };

  /// Secret assignment operator used by low-level operations such as jit tracing
  virtual void assign(const Tensor & val, RawAssignment key) = 0;

  /// Whether the variable has non-zero derivative with respect to another variable
  bool has_derivative(const VariableName & vname) const;

  /// Whether the variable has non-zero second derivative with respect to another variable
  bool has_derivative(const VariableName & v1name, const VariableName & v2name) const;

  /// Wrapper for assigning partial derivative
  Derivative<1> & d(const VariableBase & var);
  const Derivative<1> & d(const VariableBase & var) const;

  /// Wrapper for assigning second partial derivative
  Derivative<2> & d(const VariableBase & var1, const VariableBase & var2);
  const Derivative<2> & d(const VariableBase & var1, const VariableBase & var2) const;

  ///@{
  /// Request to use AD to calculate the derivative of this variable with respect to another variable
  void request_AD(const VariableBase & u);
  void request_AD(const std::vector<const VariableBase *> & us);
  ///@}

  ///@{
  /// Request to use AD to calculate the second derivative of this variable with respect to two other variables
  void request_AD(const VariableBase & u1, const VariableBase & u2);
  void request_AD(const std::vector<const VariableBase *> & u1s,
                  const std::vector<const VariableBase *> & u2s);
  ///@}

  /// Partial derivatives
  const std::vector<Derivative<1>> & derivatives() const { return _derivs; }
  std::vector<Derivative<1>> & derivatives() { return _derivs; }

  /// Partial second derivatives
  const std::vector<Derivative<2>> & second_derivatives() const { return _sec_derivs; }
  std::vector<Derivative<2>> & second_derivatives() { return _sec_derivs; }

  /// Clear the variable value and derivatives
  virtual void clear();

  /// Clear only the derivatives
  void clear_derivatives();

  /// Apply first order chain rule
  void apply_chain_rule(const DependencyResolver<Model, VariableName> &);

  /// Apply second order chain rule
  void apply_second_order_chain_rule(const DependencyResolver<Model, VariableName> &);

  /// Name of the variable
  const VariableName _name = {};

  /// The model which declared this variable
  Model * const _owner = nullptr;

private:
  ValueMap total_derivatives(const DependencyResolver<Model, VariableName> & dep,
                             Model * model,
                             const VariableBase & yvar) const;

  DerivMap total_second_derivatives(const DependencyResolver<Model, VariableName> & dep,
                                    Model * model,
                                    const VariableBase & yvar) const;

  /// Left-batch shape of the variable
  const TensorShape _lbatch_sizes = {};

  /// Derivatives of this variable with respect to other variables
  std::vector<Derivative<1>> _derivs;

  /// Second derivatives of this variable with respect to other variables
  std::vector<Derivative<2>> _sec_derivs;
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
  Variable(VariableName name_in, Model * owner, TensorShapeRef lbatch_shape)
    : VariableBase(std::move(name_in), owner, lbatch_shape),
      _base_sizes(T::const_base_sizes),
      _ref(nullptr),
      _ref_is_mutable(false)
  {
  }

  template <typename T2 = T, typename = typename std::enable_if_t<std::is_same_v<Tensor, T2>>>
  Variable(VariableName name_in,
           Model * owner,
           TensorShapeRef lbatch_shape,
           TensorShapeRef base_shape)
    : VariableBase(std::move(name_in), owner, lbatch_shape),
      _base_sizes(base_shape),
      _ref(nullptr),
      _ref_is_mutable(false)
  {
  }

  TensorType type() const override;

  TensorShapeRef base_sizes() const override { return _base_sizes; }

  std::unique_ptr<VariableBase> clone(const VariableName & name = {},
                                      Model * owner = nullptr) const override;

  void ref(const VariableBase & var, bool ref_is_mutable = false) override;

  const VariableBase * ref() const override { return _ref ? _ref->ref() : this; }

  bool owning() const override { return !_ref; }

  void zero(const TensorOptions & options) override;

  void set(const Tensor & val) override;

  Tensor get() const override;

  Tensor tensor() const override;

  void requires_grad_(bool req = true) override;

  void operator=(const Tensor & val) override;

  void assign(const Tensor & val, RawAssignment key) override;

  /// Variable value
  const T & value() const { return owning() ? _value : _ref->value(); }

  /// Negation
  T operator-() const { return -value(); }

  /// Convert to the underlying tensor type
  operator T() const { return value(); }

  void clear() override;

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

/// Derivative wrapper
template <std::size_t N>
class Derivative
{
public:
  Derivative() = default;
  Derivative(const VariableBase * var, const std::array<const VariableBase *, N> & args);

  /// Assignment operator
  ///@{
  Derivative & operator=(const Tensor & val);
  Derivative & operator=(const VariableBase & val);
  ///@}

  /// Equality operator (for searching in a container)
  bool operator==(const Derivative & other) const;

  /// Get the derivative
  const Tensor & tensor() const { return _deriv; }

  /// Get the derivative in assembly format
  const Tensor & get() const;

  /// Set the derivative (given in assembly format)
  void set(const Tensor & val);

  /// Get the args
  const std::array<const VariableBase *, N> & args() const { return _args; }

private:
  /// Variable to take derivative with
  const VariableBase * _var = nullptr;

  /// Arguments
  std::array<const VariableBase *, N> _args;

  /// Derivative to write to
  Tensor _deriv;

  /// Derivative in assembly format
  mutable Tensor _deriv_assembly;

#ifndef NDEBUG
  const std::string _debug_name = "";
#endif
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
