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
VariableStore::VariableStore(OptionSet options, Model * object)
  : _object(object),
    _object_options(std::move(options)),
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
  neml_assert(var_ptr, "Input variable ", name, " does not exist in model ", _object->name());
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
VariableStore::assign_input(const ValueMap & vals)
{
  for (const auto & [name, val] : vals)
    if (input_axis().has_variable(name))
      input_variable(name).set(val.clone());
}

void
VariableStore::assign_output(const ValueMap & vals)
{
  for (const auto & [name, val] : vals)
    output_variable(name).set(val.clone());
}

void
VariableStore::assign_output_derivatives(const DerivMap & derivs)
{
  for (const auto & [yvar, deriv] : derivs)
  {
    auto & y = output_variable(yvar);
    for (const auto & [xvar, val] : deriv)
      y.derivatives().insert_or_assign(xvar, val.clone());
  }
}

ValueMap
VariableStore::collect_input() const
{
  ValueMap vals;
  for (auto && [name, var] : input_variables())
    vals[name] = var.tensor();
  return vals;
}

ValueMap
VariableStore::collect_output() const
{
  ValueMap vals;
  for (auto && [name, var] : output_variables())
    vals[name] = var.tensor();
  return vals;
}

DerivMap
VariableStore::collect_output_derivatives() const
{
  DerivMap derivs;
  for (auto && [name, var] : output_variables())
    derivs[name] = var.derivatives();
  return derivs;
}

SecDerivMap
VariableStore::collect_output_second_derivatives() const
{
  SecDerivMap sec_derivs;
  for (auto && [name, var] : output_variables())
    sec_derivs[name] = var.second_derivatives();
  return sec_derivs;
}

} // namespace neml2
