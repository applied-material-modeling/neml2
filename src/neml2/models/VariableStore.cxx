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
    _output_axis(declare_axis("output"))
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
VariableStore::variable(const VariableName & name)
{
  auto var_ptr = _variables.query_value(name);
  neml_assert(var_ptr, "Variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

const VariableBase &
VariableStore::variable(const VariableName & name) const
{
  const auto var_ptr = _variables.query_value(name);
  neml_assert(var_ptr, "Variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

std::vector<const VariableBase *>
VariableStore::variables(FType ft) const
{
  std::vector<const VariableBase *> vars;
  for (auto && [name, var] : variables())
    if ((var.ftype() & ft) != FType::NONE)
      vars.push_back(&var);
  return vars;
}

std::vector<VariableBase *>
VariableStore::variables(FType ft)
{
  std::vector<VariableBase *> vars;
  for (auto && [name, var] : variables())
    if ((var.ftype() & ft) != FType::NONE)
      vars.push_back(&var);
  return vars;
}

void
VariableStore::clear()
{
  for (auto && [name, var] : variables())
    var.clear();
}

void
VariableStore::assign_values(const LabeledVector & vals)
{
  // The vector could be empty. Let's be lenient in this case, as this is a no-op in the worst case.
  if (vals.base_size(0) == 0)
    return;

  for (const auto & [var, val] : vals.split_variables(/*qualified=*/true))
    variable(var).set(val);
}

void
VariableStore::assign_derivatives(const LabeledMatrix & derivs)
{
  // The matrix could be empty. Let's be lenient in this case, as this is a no-op in the worst case.
  if (derivs.base_size(0) == 0 || derivs.base_size(1) == 0)
    return;

  for (const auto & [yvar, deriv] : derivs.disassemble_variables(/*qualified=*/true))
    variable(yvar).derivatives().insert(deriv.begin(), deriv.end());
}

LabeledVector
VariableStore::assemble_values(const LabeledAxis & axis) const
{
  neml_assert_dbg(axis.storage_size(), "Cannot assemble values for an empty axis.");

  auto vars = axis.qualified_variable_names();
  auto vals_flat = std::vector<Tensor>(vars.size());
  for (std::size_t i = 0; i < vars.size(); ++i)
    vals_flat[i] = variable(vars[i]).get().base_flatten();
  return LabeledVector::assemble(vals_flat, axis);
}

LabeledMatrix
VariableStore::assemble_derivatives(const LabeledAxis & yaxis, const LabeledAxis & xaxis) const
{
  neml_assert_dbg(yaxis.storage_size(), "Cannot assemble derivatives for an empty yaxis.");
  neml_assert_dbg(xaxis.storage_size(), "Cannot assemble derivatives for an empty xaxis.");

  auto yvars = yaxis.qualified_variable_names();
  auto xvars = xaxis.qualified_variable_names();
  auto vals_flat = std::vector<std::vector<Tensor>>(yvars.size());
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto & derivs = variable(yvars[i]).derivatives();
    vals_flat[i].resize(xvars.size());
    for (std::size_t j = 0; j < xvars.size(); ++j)
      if (derivs.count(xvars[j]))
        vals_flat[i][j] = derivs.at(xvars[j]);
  }
  return LabeledMatrix::assemble(vals_flat, yaxis, xaxis);
}

LabeledTensor3D
VariableStore::assemble_second_derivatives(const LabeledAxis & yaxis,
                                           const LabeledAxis & x1axis,
                                           const LabeledAxis & x2axis) const
{
  neml_assert_dbg(yaxis.storage_size(), "Cannot assemble derivatives for an empty yaxis.");
  neml_assert_dbg(x1axis.storage_size(), "Cannot assemble derivatives for an empty x1axis.");
  neml_assert_dbg(x2axis.storage_size(), "Cannot assemble derivatives for an empty x2axis.");

  auto yvars = yaxis.qualified_variable_names();
  auto x1vars = x1axis.qualified_variable_names();
  auto x2vars = x2axis.qualified_variable_names();
  auto vals_flat = std::vector<std::vector<std::vector<Tensor>>>(yvars.size());
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto & secderivs = variable(yvars[i]).second_derivatives();
    vals_flat[i].resize(x1vars.size());
    for (std::size_t j = 0; j < x1vars.size(); ++j)
      if (secderivs.count(x1vars[j]))
      {
        vals_flat[i][j].resize(x2vars.size());
        for (std::size_t k = 0; k < x2vars.size(); ++k)
          if (secderivs.at(x1vars[j]).count(x2vars[k]))
            vals_flat[i][j][k] = secderivs.at(x1vars[j]).at(x2vars[k]);
      }
  }
  return LabeledTensor3D::assemble(vals_flat, yaxis, x1axis, x2axis);
}
} // namespace neml2
