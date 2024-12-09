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

#include "neml2/models/VariableStore.h"
#include "neml2/models/Model.h"

namespace neml2
{
VariableStore::VariableStore(const OptionSet & options, Model * object)
  : _object(object),
    _object_options(options),
    _input_axis(declare_axis("input")),
    _output_axis(declare_axis("output")),
    _tensor_options(default_tensor_options())
{
}

LabeledAxis &
VariableStore::declare_axis(const std::string & name)
{
  neml_assert(!_axes.has_key(name),
              "Trying to declare an axis named ",
              name,
              ", but an axis with the same name already exists.");

  auto axis = std::make_unique<LabeledAxis>();
  return *_axes.set_pointer(name, std::move(axis));
}

void
VariableStore::setup_layout()
{
  input_axis().setup_layout();
  output_axis().setup_layout();
}

VariableBase &
VariableStore::input_variable(const VariableName & name)
{
  auto * var_ptr = _input_variables.query_value(name);
  neml_assert(var_ptr, "Input variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

const VariableBase &
VariableStore::input_variable(const VariableName & name) const
{
  const auto * var_ptr = _input_variables.query_value(name);
  neml_assert(var_ptr, "INput variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

VariableBase &
VariableStore::output_variable(const VariableName & name)
{
  auto * var_ptr = _output_variables.query_value(name);
  neml_assert(var_ptr, "Output variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

const VariableBase &
VariableStore::output_variable(const VariableName & name) const
{
  const auto * var_ptr = _output_variables.query_value(name);
  neml_assert(var_ptr, "Output variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

void
VariableStore::clear_input()
{
  for (auto && [name, var] : input_variables())
    if (var.owning())
      var.clear();
}

void
VariableStore::clear_output()
{
  for (auto && [name, var] : output_variables())
    if (var.owning())
      var.clear();
}

void
VariableStore::zero_input()
{
  for (auto && [name, var] : input_variables())
    if (var.owning())
      var.zero(_tensor_options);
}

void
VariableStore::zero_output()
{
  for (auto && [name, var] : output_variables())
    if (var.owning())
      var.zero(_tensor_options);
}

void
VariableStore::assign_input(const std::map<VariableName, Tensor> & vals)
{
  for (const auto & [name, val] : vals)
    input_variable(name).set(val.clone());
}

void
VariableStore::assign_output(const std::map<VariableName, Tensor> & vals)
{
  for (const auto & [name, val] : vals)
    output_variable(name).set(val.clone());
}

void
VariableStore::assign_output_derivatives(
    const std::map<VariableName, std::map<VariableName, Tensor>> & derivs)
{
  for (const auto & [yvar, deriv] : derivs)
    output_variable(yvar).derivatives().insert(deriv.begin(), deriv.end());
}

std::map<VariableName, Tensor>
VariableStore::collect_input() const
{
  std::map<VariableName, Tensor> vals;
  for (auto && [name, var] : input_variables())
    vals[name] = var.tensor();
  return vals;
}

std::map<VariableName, Tensor>
VariableStore::collect_output() const
{
  std::map<VariableName, Tensor> vals;
  for (auto && [name, var] : output_variables())
    vals[name] = var.tensor();
  return vals;
}

std::map<VariableName, std::map<VariableName, Tensor>>
VariableStore::collect_output_derivatives() const
{
  std::map<VariableName, std::map<VariableName, Tensor>> derivs;
  for (auto && [name, var] : output_variables())
    derivs[name] = var.derivatives();
  return derivs;
}

std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>>
VariableStore::collect_output_second_derivatives() const
{
  std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>> sec_derivs;
  for (auto && [name, var] : output_variables())
    sec_derivs[name] = var.second_derivatives();
  return sec_derivs;
}

} // namespace neml2
