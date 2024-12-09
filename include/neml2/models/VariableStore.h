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

#include <ATen/core/stack.h>

#include "neml2/base/NEML2Object.h"
#include "neml2/base/Storage.h"
#include "neml2/models/LabeledAxis.h"
#include "neml2/models/Variable.h"
#include "neml2/models/map_types.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
// Foward decl
class Model;

class VariableStore
{
public:
  VariableStore(OptionSet options, Model * object);

  VariableStore(const VariableStore &) = delete;
  VariableStore(VariableStore &&) = delete;
  VariableStore & operator=(const VariableStore &) = delete;
  VariableStore & operator=(VariableStore &&) = delete;
  virtual ~VariableStore() = default;

  LabeledAxis & declare_axis(const std::string & name);

  /// Setup the layout of all the registered axes
  virtual void setup_layout();

  ///@{
  /// Definition of the input axis showing the layout of input variables
  LabeledAxis & input_axis() { return _input_axis; }
  const LabeledAxis & input_axis() const { return _input_axis; }
  ///@}

  ///@{
  /// Definition of the output axis showing the layout of output variables
  LabeledAxis & output_axis() { return _output_axis; }
  const LabeledAxis & output_axis() const { return _output_axis; }
  ///@}

  ///@{
  /// Variables
  Storage<VariableName, VariableBase> & input_variables() { return _input_variables; }
  const Storage<VariableName, VariableBase> & input_variables() const { return _input_variables; }
  Storage<VariableName, VariableBase> & output_variables() { return _output_variables; }
  const Storage<VariableName, VariableBase> & output_variables() const { return _output_variables; }
  ///@}

  ///@{
  /// Lookup a variable by name
  VariableBase & input_variable(const VariableName &);
  const VariableBase & input_variable(const VariableName &) const;
  VariableBase & output_variable(const VariableName &);
  const VariableBase & output_variable(const VariableName &) const;
  ///@}

  /// Current tensor options
  const torch::TensorOptions & tensor_options() const { return _tensor_options; }

  ///@{
  /// Release allocated tensor
  virtual void clear_input();
  virtual void clear_output();
  ///@}

  ///@{
  /// Zero variable values
  virtual void zero_input();
  virtual void zero_output();
  ///@}

  ///@{
  /// Assign input variable values
  void assign_input(const ValueMap & vals);
  /// Assign output variable values
  void assign_output(const ValueMap & vals);
  /// Assign variable derivatives
  void
  assign_output_derivatives(const std::map<VariableName, std::map<VariableName, Tensor>> & derivs);
  ///@}

  ///@{
  /// Collect variable values
  ValueMap collect_input() const;
  ValueMap collect_output() const;
  /// Collect variable derivatives
  DerivMap collect_output_derivatives() const;
  /// Collect variable second derivatives
  SecDerivMap collect_output_second_derivatives() const;
  ///@}

protected:
  /// Declare an input variable
  template <typename T, typename S>
  const Variable<T> &
  declare_input_variable(S && name, TensorShapeRef list_shape = {}, TensorShapeRef base_shape = {})
  {
    if constexpr (!std::is_same_v<T, Tensor>)
      neml_assert(base_shape.empty(),
                  "Creating a Variable of primitive tensor type does not require a base shape.");

    const auto var_name = variable_name(std::forward<S>(name));
    const auto list_sz = utils::storage_size(list_shape);
    const auto base_sz =
        std::is_same_v<T, Tensor> ? utils::storage_size(base_shape) : T::const_base_storage;
    const auto sz = list_sz * base_sz;

    _input_axis.add_variable(var_name, sz);
    return *create_variable<T>(_input_variables, var_name, list_shape, base_shape);
  }

  /// Declare an output variable
  template <typename T, typename S>
  Variable<T> &
  declare_output_variable(S && name, TensorShapeRef list_shape = {}, TensorShapeRef base_shape = {})
  {
    if constexpr (!std::is_same_v<T, Tensor>)
      neml_assert(base_shape.empty(),
                  "Creating a Variable of primitive tensor type does not require a base shape.");

    const auto var_name = variable_name(std::forward<S>(name));
    const auto list_sz = utils::storage_size(list_shape);
    const auto base_sz =
        std::is_same_v<T, Tensor> ? utils::storage_size(base_shape) : T::const_base_storage;
    const auto sz = list_sz * base_sz;

    _output_axis.add_variable(var_name, sz);
    return *create_variable<T>(_output_variables, var_name, list_shape, base_shape);
  }

  /// Clone a variable and put it on the input axis
  const VariableBase * clone_input_variable(const VariableBase & var,
                                            const VariableName & new_name = {})
  {
    neml_assert(&var.owner() != _object, "Trying to clone a variable from the same model.");

    const auto var_name = new_name.empty() ? var.name() : new_name;
    neml_assert(
        !_input_variables.query_value(var_name), "Input variable ", var_name, " already exists.");
    auto var_clone = var.clone(var_name, _object);

    _input_axis.add_variable(var_name, var_clone->assembly_storage());
    return _input_variables.set_pointer(var_name, std::move(var_clone));
  }

  /// Clone a variable and put it on the output axis
  VariableBase * clone_output_variable(const VariableBase & var, const VariableName & new_name = {})
  {
    neml_assert(&var.owner() != _object, "Trying to clone a variable from the same model.");

    const auto var_name = new_name.empty() ? var.name() : new_name;
    neml_assert(
        !_output_variables.query_value(var_name), "Output variable ", var_name, " already exists.");
    auto var_clone = var.clone(var_name, _object);

    _output_axis.add_variable(var_name, var_clone->assembly_storage());
    return _output_variables.set_pointer(var_name, std::move(var_clone));
  }

  /// Assign stack to input variables
  void assign_input_stack(torch::jit::Stack & stack);

  /// Assign stack to output variables and derivatives
  void assign_output_stack(torch::jit::Stack & stack, bool out, bool dout, bool d2out);

  /// Collect stack from input variables
  torch::jit::Stack collect_input_stack() const;

  /// Collect stack from output variables and derivatives
  torch::jit::Stack collect_output_stack(bool out, bool dout, bool d2out) const;

private:
  // Helper method to construct variable name
  template <typename S>
  VariableName variable_name(S && name) const
  {
    if constexpr (std::is_convertible_v<S, std::string>)
      if (_object_options.contains<VariableName>(name))
        return _object_options.get<VariableName>(name);

    return name;
  }

  // Create a variable
  template <typename T>
  Variable<T> * create_variable(Storage<VariableName, VariableBase> & variables,
                                const VariableName & name,
                                TensorShapeRef list_shape,
                                TensorShapeRef base_shape)
  {
    // Make sure we don't duplicate variables
    VariableBase * var_base_ptr = variables.query_value(name);
    neml_assert(!var_base_ptr,
                "Trying to create variable ",
                name,
                ", but a variable with the same name already exists.");

    // Allocate
    if constexpr (std::is_same_v<T, Tensor>)
    {
      auto var = std::make_unique<Variable<Tensor>>(name, _object, list_shape, base_shape);
      var_base_ptr = variables.set_pointer(name, std::move(var));
    }
    else
    {
      auto var = std::make_unique<Variable<T>>(name, _object, list_shape);
      var_base_ptr = variables.set_pointer(name, std::move(var));
    }

    // Cast it to the concrete type
    auto var_ptr = dynamic_cast<Variable<T> *>(var_base_ptr);
    neml_assert(
        var_ptr, "Internal error: Failed to cast variable ", name, " to its concrete type.");

    return var_ptr;
  }

  /// Model using this interface
  Model * _object;

  /**
   * @brief Parsed input file options for this object.

   * These options are useful for example when we declare a variable using an input option name.
   *
   */
  const OptionSet _object_options;

  /// All the declared axes
  Storage<std::string, LabeledAxis> _axes;

  /// The input axis
  LabeledAxis & _input_axis;

  /// The output axis
  LabeledAxis & _output_axis;

  /// Input variables
  Storage<VariableName, VariableBase> _input_variables;

  /// Output variables
  Storage<VariableName, VariableBase> _output_variables;

  /// Current tensor options
  torch::TensorOptions _tensor_options;
};
} // namespace neml2
