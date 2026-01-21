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
#include "neml2/models/utils.h"
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

  /**
   * @brief The canonical constructor
   *
   * @param name_in Variable name
   * @param owner Model who declared this variable
   * @param base_shape Base shape of this variable
   */
  VariableBase(VariableName name_in, Model * owner, TensorShapeRef base_shape);

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
  virtual bool defined() const = 0;
  /// Tensor options
  virtual TensorOptions options() const = 0;
  /// Scalar type
  virtual Dtype scalar_type() const = 0;
  /// Device
  virtual Device device() const = 0;
  ///@}

  /// @return the number of dimensions
  ///@{
  Size dim() const;
  Size batch_dim() const;
  Size base_dim() const;
  Size dynamic_dim() const;
  Size static_dim() const;
  Size intmd_dim() const;
  ///@}

  /// @return the tensor shape
  ///@{
  TensorShapeRef sizes() const;
  TraceableTensorShape batch_sizes() const;
  TensorShapeRef base_sizes() const;
  virtual const TraceableTensorShape & dynamic_sizes() const = 0;
  TensorShapeRef static_sizes() const;
  TensorShapeRef intmd_sizes() const;
  ///@}

  /// @return the size of dimension @p i
  ///@{
  Size size(Size i) const;
  TraceableSize batch_size(Size i) const;
  Size base_size(Size i) const;
  const TraceableSize & dynamic_size(Size i) const;
  Size static_size(Size i) const;
  Size intmd_size(Size i) const;
  ///@}

  /// Clone this variable
  virtual std::unique_ptr<VariableBase> clone(const VariableName & name = {},
                                              Model * owner = nullptr) const = 0;

  /// Reference another variable
  virtual void ref(VariableBase & other) = 0;

  /// Get the referencing variable (returns this if this is a storing variable)
  virtual const VariableBase * ref() const = 0;
  virtual VariableBase * ref() = 0;

  /// Get the direct referencing variable (returns nullptr if this is a storing variable)
  virtual const VariableBase * direct_ref() const = 0;
  virtual VariableBase * direct_ref() = 0;

  /// Check if this is an owning variable
  virtual bool owning() const = 0;

  /// Whether this variable is mutable when it is referenced by another variable
  bool is_mutable() const;

  /// Allow/disable mutation of this variable when it is referenced by another variable
  void set_mutable(bool m);

  /// Make zeros tensor with the shape of this variable
  Tensor zeros(const TensorOptions & options) const;

  /// Set the variable value to zero
  virtual void zero(const TensorOptions & options) = 0;

  /// Get the variable value cast to Tensor
  virtual Tensor tensor() const = 0;

  /// Check if this variable is part of the AD function graph
  bool requires_grad() const;

  /// Mark this variable as a leaf variable in tracing function graph for AD
  virtual void requires_grad_(bool req = true) = 0;

  /// Assignment operator (with TracerPrivilege)
  virtual void assign(const Tensor & val, std::optional<TracerPrivilege> key = std::nullopt) = 0;

  /// Assignment operator
  virtual void operator=(const Tensor & val) = 0;

  /// Whether the variable has non-zero derivative with respect to another variable
  bool has_derivative(const VariableName & vname) const;

  /// Whether the variable has non-zero second derivative with respect to another variable
  bool has_derivative(const VariableName & v1name, const VariableName & v2name) const;

  /// Wrapper for assigning partial derivative
  Derivative<1> & d(const VariableBase & arg,
                    std::size_t deriv_intrsc_intmd_dim = 0,
                    std::size_t var_intrsc_intmd_dim = 0,
                    std::size_t arg_intrsc_intmd_dim = 0);
  const Derivative<1> & d(const VariableBase & arg) const;

  /// Wrapper for assigning second partial derivative
  Derivative<2> & d2(const VariableBase & arg1,
                     const VariableBase & arg2,
                     std::size_t deriv_intrsc_intmd_dim = 0,
                     std::size_t var_intrsc_intmd_dim = 0,
                     std::size_t arg1_intrsc_intmd_dim = 0,
                     std::size_t arg2_intrsc_intmd_dim = 0);
  const Derivative<2> & d2(const VariableBase & arg1, const VariableBase & arg2) const;
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

  using DerivTuple = std::tuple<Derivative<1>, const VariableBase *>;
  using DerivContainer = std::vector<DerivTuple>;
  using SecDerivTuple = std::tuple<Derivative<2>, const VariableBase *, const VariableBase *>;
  using SecDerivContainer = std::vector<SecDerivTuple>;

  /// Partial derivatives
  const DerivContainer & derivatives() const { return _derivs; }
  DerivContainer & derivatives() { return _derivs; }

  /// Partial second derivatives
  const SecDerivContainer & second_derivatives() const { return _sec_derivs; }
  SecDerivContainer & second_derivatives() { return _sec_derivs; }

  /// Clear the variable value and derivatives
  virtual void clear();

  /// Clear only the derivatives
  void clear_derivatives();

  ///@{
  /// Whether this variable is a leaf variable in the dependency graph
  bool is_leaf(const DependencyResolver<Model, VariableName> &) const;
  /// Get the provider in the dependency graph
  const VariableBase & provider(const DependencyResolver<Model, VariableName> &) const;
  /// Get total derivatives with respect to leaf variables
  const DerivContainer & total_derivatives(const DependencyResolver<Model, VariableName> &) const;
  /// Get total second derivatives with respect to leaf variables
  const SecDerivContainer &
  total_second_derivatives(const DependencyResolver<Model, VariableName> &) const;
  /// Clear chain rule cache
  void clear_chain_rule_cache(const DependencyResolver<Model, VariableName> &) const;
  ///@}

protected:
  /// Name of the variable
  const VariableName _name = {};

  /// The model which declared this variable
  Model * const _owner = nullptr;

  /// Cached intermediate shape that this variable last saw
  /// @note: set() and operator=() are the only methods that cache this. clear() does not invalidate the cache.
  TensorShape _cached_intmd_sizes = {};

  /// Base shape of the variable
  const TensorShape _base_sizes = {};

  /// When referenced by another variable, whether to allow the referencing variable to mutate my value
  bool _mutable = false;

private:
  ///@{
  /// Derivatives of this variable with respect to other variables
  DerivContainer _derivs;
  /// Second derivatives of this variable with respect to other variables
  SecDerivContainer _sec_derivs;
  /// Cache for total derivatives (with respect to leaf variables)
  mutable DerivContainer _total_derivs;
  /// Cache for second total derivatives (with respect to leaf variables)
  mutable SecDerivContainer _total_sec_derivs;
  ///@}
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
      return op(a(), b());                                                                         \
                                                                                                   \
    if constexpr (std::is_base_of_v<VariableBase, T1> && !std::is_base_of_v<VariableBase, T2>)     \
      return op(a(), b);                                                                           \
                                                                                                   \
    if constexpr (!std::is_base_of_v<VariableBase, T1> && std::is_base_of_v<VariableBase, T2>)     \
      return op(a, b());                                                                           \
  }                                                                                                \
  static_assert(true)

FWD_VARIABLE_BINARY_OP(operator+);
FWD_VARIABLE_BINARY_OP(operator-);
FWD_VARIABLE_BINARY_OP(operator*);
FWD_VARIABLE_BINARY_OP(operator/);

FWD_VARIABLE_BINARY_OP(operator>);
FWD_VARIABLE_BINARY_OP(operator<);
FWD_VARIABLE_BINARY_OP(operator>=);
FWD_VARIABLE_BINARY_OP(operator<=);
FWD_VARIABLE_BINARY_OP(operator&&);
FWD_VARIABLE_BINARY_OP(operator||);
FWD_VARIABLE_BINARY_OP(operator==);
FWD_VARIABLE_BINARY_OP(operator!=);
}
