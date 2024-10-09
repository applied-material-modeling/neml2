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

#include "neml2/base/NEML2Object.h"
#include "neml2/base/Storage.h"
#include "neml2/tensors/Variable.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
// Foward decl
class Model;

class VariableStore
{
public:
  VariableStore(const OptionSet & options, Model * object);

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
  Storage<VariableName, VariableBase> & variables() { return _variables; }
  const Storage<VariableName, VariableBase> & variables() const { return _variables; }
  ///@}

  ///@{
  /// Lookup a variable by name
  VariableBase & variable(const VariableName &);
  const VariableBase & variable(const VariableName &) const;
  ///@}

  /// List of variables that match the given ftype
  std::vector<const VariableBase *> variables(FType) const;
  std::vector<VariableBase *> variables(FType);

  /// Assign variable values
  void assign_values(const LabeledVector & vals);

  /// Assign derivatives
  void assign_derivatives(const LabeledMatrix & derivs);

  /// Assemble (flattened) variable values into a vector
  LabeledVector assemble_values(const LabeledAxis & axis) const;

  /// Assemble derivatives into a matrix
  LabeledMatrix assemble_derivatives(const LabeledAxis & yaxis, const LabeledAxis & xaxis) const;

  /// Assemble second derivatives into a 3D tensor
  LabeledTensor3D assemble_second_derivatives(const LabeledAxis & yaxis,
                                              const LabeledAxis & x1axis,
                                              const LabeledAxis & x2axis) const;

protected:
  /// Cleanup after evaluation
  virtual void clear();

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

    _input_axis.add(var_name, sz);
    return *create_variable<T>(var_name, FType::INPUT, list_shape, base_shape);
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

    _output_axis.add(var_name, sz);
    return *create_variable<T>(var_name, FType::OUTPUT, list_shape, base_shape);
  }

  /// Clone a variable and put it on the input axis
  const VariableBase * clone_input_variable(const VariableBase & var,
                                            const VariableName & new_name = {})
  {
    neml_assert(&var.owner() != _object, "Trying to clone a variable from the same model.");

    const auto var_name = new_name.empty() ? var.name() : new_name;
    const auto * var_exist = _variables.query_value(var_name);
    const auto ftype = var_exist ? (var_exist->ftype() | FType::INPUT) : FType::INPUT;
    auto var_clone = var.clone(var_name, _object, ftype);

    _input_axis.add(var_name, var_clone->assembly_storage());
    return _variables.set_pointer(var_name, std::move(var_clone));
  }

  /// Clone a variable and put it on the output axis
  VariableBase * clone_output_variable(const VariableBase & var, const VariableName & new_name = {})
  {
    neml_assert(&var.owner() != _object, "Trying to clone a variable from the same model.");

    const auto var_name = new_name.empty() ? var.name() : new_name;
    const auto * var_exist = _variables.query_value(var_name);
    const auto ftype = var_exist ? (var_exist->ftype() | FType::OUTPUT) : FType::OUTPUT;
    auto var_clone = var.clone(var_name, _object, ftype);

    _output_axis.add(var_name, var_clone->assembly_storage());
    return _variables.set_pointer(var_name, std::move(var_clone));
  }

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
  Variable<T> * create_variable(const VariableName & name,
                                FType ft,
                                TensorShapeRef list_shape,
                                TensorShapeRef base_shape)
  {
    // Make sure we don't duplicate variables
    VariableBase * var_base_ptr = _variables.query_value(name);
    neml_assert(!var_base_ptr,
                "Trying to create variable ",
                name,
                ", but a variable with the same name already exists.");

    // Allocate
    if constexpr (std::is_same_v<T, Tensor>)
    {
      auto var = std::make_unique<Variable<Tensor>>(name, _object, ft, list_shape, base_shape);
      var_base_ptr = _variables.set_pointer(name, std::move(var));
    }
    else
    {
      auto var = std::make_unique<Variable<T>>(name, _object, ft, list_shape);
      var_base_ptr = _variables.set_pointer(name, std::move(var));
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

  /// Variables
  Storage<VariableName, VariableBase> _variables;
};
} // namespace neml2
