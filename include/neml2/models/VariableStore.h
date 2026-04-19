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

#include <memory>
#include <optional>

#include "neml2/misc/types.h"
#include "neml2/tensors/jit.h"
#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/equation_systems/SparseVector.h"

namespace neml2
{
// Foward declarations
class Model;
class VariableBase;
template <typename T>
class Variable;
template <typename T>
struct TensorName;

class VariableStore
{
public:
  VariableStore(Model * object);

  VariableStore(const VariableStore &) = delete;
  VariableStore(VariableStore &&) = delete;
  VariableStore & operator=(const VariableStore &) = delete;
  VariableStore & operator=(VariableStore &&) = delete;
  virtual ~VariableStore() = default;

  using VariableStorage = std::map<VariableName, std::unique_ptr<VariableBase>>;
  using DerivSparsity = std::vector<std::pair<VariableBase *, const VariableBase *>>;
  using SecDerivSparsity =
      std::vector<std::tuple<VariableBase *, const VariableBase *, const VariableBase *>>;

  ///@{
  /// Variables
  VariableStorage & input_variables() { return _input_variables; }
  const VariableStorage & input_variables() const { return _input_variables; }
  VariableStorage & output_variables() { return _output_variables; }
  const VariableStorage & output_variables() const { return _output_variables; }
  ///@}

  ///@{
  /// Lookup a variable by name
  VariableBase & input_variable(const VariableName &);
  const VariableBase & input_variable(const VariableName &) const;
  VariableBase & output_variable(const VariableName &);
  const VariableBase & output_variable(const VariableName &) const;
  ///@}

  /// Current tensor options for variables
  const TensorOptions & variable_options() const { return _options; }

  ///@{
  /// Release allocated tensor
  virtual void clear_input();
  virtual void clear_output();
  virtual void clear_derivatives();
  ///@}

  /// Fill undefined input variables with zeros
  virtual void zero_undefined_input();

  ///@{
  /// Cache sparsity of first derivatives
  void cache_derivative_sparsity();
  /// Cache sparsity of second derivatives
  void cache_second_derivative_sparsity();
  /// Derivative sparsity
  const std::optional<DerivSparsity> & derivative_sparsity() const;
  /// Second derivative sparsity
  const std::optional<SecDerivSparsity> & second_derivative_sparsity() const;
  ///@}

  ///@{
  /// Assign input variable values
  void assign_input(const ValueMap &, bool allow_nonexistent = false);
  void assign_input(const SparseVector &, bool allow_nonexistent = false);
  /// Assign output variable values
  void assign_output(const ValueMap &);
  void assign_output(const SparseVector &);
  /// Assign variable derivatives
  void assign_output_derivatives(const DerivMap & derivs);
  void assign_output_derivatives(const SparseMatrix &);
  ///@}

  ///@{
  /// Collect input variable values
  ValueMap collect_input() const;
  SparseVector collect_input(const AxisLayout &) const;
  /// Collect output variable values
  ValueMap collect_output() const;
  SparseVector collect_output(const AxisLayout &) const;
  /// Collect variable derivatives
  DerivMap collect_output_derivatives() const;
  SparseMatrix collect_output_derivatives(const AxisLayout &, const AxisLayout &) const;
  /// Collect variable second derivatives
  SecDerivMap collect_output_second_derivatives() const;
  ///@}

protected:
  /**
   * @brief Send padding variables to options
   *
   * @param options The target options
   */
  virtual void send_variables_to(const TensorOptions & options);

  /**
   * @brief Helper method to wrap a variable name into its rate form
   *
   * This is a personal preference in the end. The idea is to have a consistent way to name the rate
   * form of a variable, which is commonly used in time integration. The behavior can be controlled
   * via Settings::rate_prefix and Settings::rate_suffix.
   *
   * @param var The variable name
   * @return The rate form of the variable name
   */
  VariableName rate_name(const VariableName & var) const;

  /**
   * @brief Helper method to wrap a variable name into its residual form
   *
   * Similar to rate_name, this is for consistent naming of residual variables, which are commonly
   * used in nonlinear systems. The behavior can be controlled via Settings::residual_prefix and
   * Settings::residual_suffix.
   *
   * @param var The variable name
   * @return The residual form of the variable name
   */
  VariableName residual_name(const VariableName & var) const;

  /**
   * @brief Declare an input variable
   *
   * @tparam T Tensor type
   * @param name Variable name
   * @param allow_duplicate Whether to allow duplicate variable declaration
   * @return const Variable<T>&
   */
  template <typename T>
  const Variable<T> & declare_input_variable(const char * name, bool allow_duplicate = false);

  /**
   * @brief Declare an input variable
   *
   * @tparam T Tensor type
   * @param name Variable name
   * @param allow_duplicate Whether to allow duplicate variable declaration
   * @return const Variable<T>&
   */
  template <typename T>
  const Variable<T> & declare_input_variable(const VariableName & name,
                                             bool allow_duplicate = false);

  /**
   * @brief Declare an output variable
   *
   * @tparam T Tensor type
   * @param name Variable name
   * @return const Variable<T>&
   */
  template <typename T>
  Variable<T> & declare_output_variable(const char * name);

  /**
   * @brief Declare an output variable
   *
   * @tparam T Tensor type
   * @param name Variable name
   * @return const Variable<T>&
   */
  template <typename T>
  Variable<T> & declare_output_variable(const VariableName & name);

  /// Declare a variable that holds the value of @p var from @p nstep steps back.
  template <typename T>
  const Variable<T> & declare_variable_history(const Variable<T> & var, std::size_t nstep);

  /// Clone a variable and put it on the input axis
  const VariableBase * clone_input_variable(const VariableBase & var,
                                            std::optional<VariableName> new_name = std::nullopt);

  /// Clone a variable and put it on the output axis
  VariableBase * clone_output_variable(const VariableBase & var,
                                       std::optional<VariableName> new_name = std::nullopt);

  /// Assign stack to input variables
  void assign_input_stack(jit::Stack & stack);

  /// Assign stack to output variables and derivatives
  void assign_output_stack(jit::Stack & stack, bool out, bool dout, bool d2out);

  /// Collect stack from input variables
  jit::Stack collect_input_stack() const;

  /// Collect stack from output variables and derivatives
  jit::Stack collect_output_stack(bool out, bool dout, bool d2out) const;

  // TensorName resolution may require declare_input_variable
  template <typename T>
  friend const T & resolve_tensor_name(const TensorName<T> &, Model *, const std::string &);

private:
  // Create a variable
  template <typename T>
  Variable<T> * create_variable(VariableStorage & variables,
                                const VariableName & name,
                                bool allow_duplicate = false);

  /// Model using this interface
  Model * _object;

  /// Input variables
  VariableStorage _input_variables;

  /// Output variables
  VariableStorage _output_variables;

  /// Variable histories
  std::vector<VariableStorage> _histories;

  /// Current tensor options for padding variables
  TensorOptions _options;

  /// Cached intermediate dimensions of input variables
  std::map<VariableName, Size> _input_intmd_dims;

  /// Cached intermediate dimensions of output variables
  std::map<VariableName, Size> _output_intmd_dims;

  /// Derivative sparsity
  std::optional<DerivSparsity> _deriv_sparsity = std::nullopt;

  /// Second derivative sparsity
  std::optional<SecDerivSparsity> _secderiv_sparsity = std::nullopt;

  /// Derivative sparsity for the nonlinear system
  std::optional<DerivSparsity> _deriv_sparsity_nl_sys = std::nullopt;

  /// Second derivative sparsity for the nonlinear system
  std::optional<SecDerivSparsity> _secderiv_sparsity_nl_sys = std::nullopt;
};
} // namespace neml2
