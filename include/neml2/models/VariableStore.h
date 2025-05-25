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

#include "neml2/models/map_types_fwd.h"
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/misc/types.h"
#include "neml2/jit/types.h"

namespace neml2
{
// Foward declarations
class Model;
class LabeledAxis;
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

  using VariableStorage = std::map<VariableName, std::unique_ptr<VariableBase>>;

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
  ///@}

  ///@{
  /// Zero variable values
  virtual void zero_input();
  virtual void zero_output();
  ///@}

  ///@{
  /// Assign input variable values
  void assign_input(const ValueMap & vals);
  /// Assign input variable values
  void assign_input(ValueMap && vals);
  /// Assign output variable values
  void assign_output(const ValueMap & vals);
  /// Assign variable derivatives
  void assign_output_derivatives(const DerivMap & derivs);
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
  /**
   * @brief Send padding variables to options
   *
   * @param options The target options
   */
  virtual void send_variables_to(const TensorOptions & options);

  /// Declare an input variable
  template <typename T>
  const Variable<T> & declare_input_variable(const char * name, TensorShapeRef list_shape = {});

  /// Declare an input variable
  template <typename T>
  const Variable<T> & declare_input_variable(const VariableName & name,
                                             TensorShapeRef list_shape = {});

  /// Declare an output variable
  template <typename T>
  Variable<T> & declare_output_variable(const char * name, TensorShapeRef list_shape = {});

  /// Declare an output variable
  template <typename T>
  Variable<T> & declare_output_variable(const VariableName & name, TensorShapeRef list_shape = {});

  /// Clone a variable and put it on the input axis
  const VariableBase * clone_input_variable(const VariableBase & var,
                                            const VariableName & new_name = {});

  /// Clone a variable and put it on the output axis
  VariableBase * clone_output_variable(const VariableBase & var,
                                       const VariableName & new_name = {});

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
                                TensorShapeRef list_shape);

  /// Model using this interface
  Model * _object;

  /// All the declared axes
  std::map<std::string, std::unique_ptr<LabeledAxis>> _axes;

  /// The input axis
  LabeledAxis & _input_axis;

  /// The output axis
  LabeledAxis & _output_axis;

  /// Input variables
  VariableStorage _input_variables;

  /// Output variables
  VariableStorage _output_variables;

  /// Current tensor options for padding variables
  TensorOptions _options;
};
} // namespace neml2
