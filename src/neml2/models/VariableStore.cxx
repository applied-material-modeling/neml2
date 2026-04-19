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
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/misc/types.h"
#include "neml2/models/Model.h"
#include "neml2/misc/assertions.h"
#include "neml2/models/Variable.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/tensors.h"
#include "neml2/base/Settings.h"

namespace neml2
{

VariableStore::VariableStore(Model * object)
  : _object(object),
    _options(default_tensor_options())
{
}

template <typename T>
const Variable<T> &
VariableStore::declare_input_variable(const char * name, bool allow_duplicate)
{
  if (_object->input_options().contains(name))
    return declare_input_variable<T>(_object->input_options().get<VariableName>(name),
                                     allow_duplicate);
  return declare_input_variable<T>(VariableName(name), allow_duplicate);
}

template <typename T>
const Variable<T> &
VariableStore::declare_input_variable(const VariableName & name, bool allow_duplicate)
{
  return *create_variable<T>(_input_variables, name, allow_duplicate);
}
#define INSTANTIATE_DECLARE_INPUT_VARIABLE(T)                                                      \
  template const Variable<T> & VariableStore::declare_input_variable<T>(const char *, bool);       \
  template const Variable<T> & VariableStore::declare_input_variable<T>(const VariableName &, bool)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_DECLARE_INPUT_VARIABLE);

template <typename T>
Variable<T> &
VariableStore::declare_output_variable(const char * name)
{
  if (_object->input_options().contains(name))
    return declare_output_variable<T>(_object->input_options().get<VariableName>(name));
  return declare_output_variable<T>(VariableName(name));
}

template <typename T>
Variable<T> &
VariableStore::declare_output_variable(const VariableName & name)
{
  return *create_variable<T>(_output_variables, name);
}
#define INSTANTIATE_DECLARE_OUTPUT_VARIABLE(T)                                                     \
  template Variable<T> & VariableStore::declare_output_variable<T>(const char *);                  \
  template Variable<T> & VariableStore::declare_output_variable<T>(const VariableName &)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_DECLARE_OUTPUT_VARIABLE);

template <typename T>
const Variable<T> &
VariableStore::declare_variable_history(const Variable<T> & var, std::size_t nstep)
{
  neml_assert(nstep > 0,
              "Trying to declare variable history for '",
              var.name(),
              "' with nstep = ",
              nstep,
              ". nstep should be positive.");
  if (nstep > _histories.size())
    _histories.resize(nstep);
  auto * var_hist = create_variable<T>(_histories[nstep - 1], var.name());
  var.register_history(var_hist, nstep);
  return *var_hist;
}
#define INSTANTIATE_DECLARE_VARIABLE_HISTORY(T)                                                    \
  template const Variable<T> & VariableStore::declare_variable_history<T>(const Variable<T> &,     \
                                                                          std::size_t)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_DECLARE_VARIABLE_HISTORY);

const VariableBase *
VariableStore::clone_input_variable(const VariableBase & var, std::optional<VariableName> new_name)
{
  neml_assert(&var.owner() != _object, "Trying to clone a variable from the same model.");

  const auto var_name = new_name.has_value() ? new_name.value() : var.name();
  neml_assert(!_input_variables.count(var_name), "Input variable '", var_name, "' already exists.");
  auto var_clone = var.clone(var_name, _object);

  auto [it, success] = _input_variables.emplace(var_name, std::move(var_clone));
  return it->second.get();
}

VariableBase *
VariableStore::clone_output_variable(const VariableBase & var, std::optional<VariableName> new_name)
{
  neml_assert(&var.owner() != _object, "Trying to clone a variable from the same model.");

  const auto var_name = new_name.has_value() ? new_name.value() : var.name();
  neml_assert(
      !_output_variables.count(var_name), "Output variable '", var_name, "' already exists.");
  auto var_clone = var.clone(var_name, _object);

  auto [it, success] = _output_variables.emplace(var_name, std::move(var_clone));
  return it->second.get();
}

template <typename T>
Variable<T> *
VariableStore::create_variable(VariableStorage & variables,
                               const VariableName & name,
                               bool allow_duplicate)
{
  // Make sure we don't duplicate variables
  if (!allow_duplicate)
    neml_assert(!variables.count(name),
                "Trying to create variable '",
                name,
                "', but a variable with the same name already exists.");

  VariableBase * var_base_ptr = nullptr;

  if (allow_duplicate && variables.count(name))
    var_base_ptr = variables[name].get();
  else
  {
    // Allocate
    std::unique_ptr<VariableBase> var;
    var = std::make_unique<Variable<T>>(name, _object);
    auto [it, success] = variables.emplace(name, std::move(var));
    var_base_ptr = it->second.get();
  }

  // Cast it to the concrete type
  auto var_ptr = dynamic_cast<Variable<T> *>(var_base_ptr);
  if (!var_ptr)
    throw NEMLException("Internal error: Failed to cast variable '" + name +
                        "' to its concrete type.");

  return var_ptr;
}
#define INSTANTIATE_CREATE_VARIABLE(T)                                                             \
  template Variable<T> * VariableStore::create_variable<T>(                                        \
      VariableStorage &, const VariableName &, bool)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_CREATE_VARIABLE);

VariableBase &
VariableStore::input_variable(const VariableName & name)
{
  auto it = _input_variables.find(name);
  neml_assert(it != _input_variables.end(),
              "Input variable ",
              name,
              " does not exist in model ",
              _object->name());
  return *it->second;
}

const VariableBase &
VariableStore::input_variable(const VariableName & name) const
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<VariableStore *>(this)->input_variable(name);
}

VariableBase &
VariableStore::output_variable(const VariableName & name)
{
  auto it = _output_variables.find(name);
  neml_assert(it != _output_variables.end(),
              "Output variable ",
              name,
              " does not exist in model ",
              _object->name());
  return *it->second;
}

const VariableBase &
VariableStore::output_variable(const VariableName & name) const
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<VariableStore *>(this)->output_variable(name);
}

void
VariableStore::send_variables_to(const TensorOptions & options)
{
  _options = options;
}

VariableName
VariableStore::rate_name(const VariableName & var_name) const
{
  return VariableName(_object->settings().rate_prefix() + var_name +
                      _object->settings().rate_suffix());
}

VariableName
VariableStore::residual_name(const VariableName & var_name) const
{
  return VariableName(_object->settings().residual_prefix() + var_name +
                      _object->settings().residual_suffix());
}

void
VariableStore::clear_input()
{
  for (auto && [name, var] : input_variables())
    if (var->owning())
      var->clear();
}

void
VariableStore::clear_output()
{
  for (auto && [name, var] : output_variables())
    if (var->owning())
      var->clear();
}

void
VariableStore::clear_derivatives()
{
  for (auto && [name, var] : output_variables())
    var->clear_derivatives();
}

void
VariableStore::zero_undefined_input()
{
  for (auto && [name, var] : input_variables())
    if (var->owning() && !var->defined())
      var->zero(_options);
}

void
VariableStore::cache_derivative_sparsity()
{
  std::vector<std::pair<VariableBase *, const VariableBase *>> sparsity;
  for (auto && [yname, yvar] : output_variables())
    for (const auto & [dy_dx, xvar] : yvar->derivatives())
      if (dy_dx.defined())
        sparsity.emplace_back(yvar.get(), xvar);

  if (currently_assembling_nonlinear_system())
    _deriv_sparsity_nl_sys = std::move(sparsity);
  else
    _deriv_sparsity = std::move(sparsity);
}

void
VariableStore::cache_second_derivative_sparsity()
{
  std::vector<std::tuple<VariableBase *, const VariableBase *, const VariableBase *>> sparsity;
  for (auto && [yname, yvar] : output_variables())
    for (const auto & [d2y_dx1dx2, x1var, x2var] : yvar->second_derivatives())
      if (d2y_dx1dx2.defined())
        sparsity.emplace_back(yvar.get(), x1var, x2var);

  if (currently_assembling_nonlinear_system())
    _secderiv_sparsity_nl_sys = std::move(sparsity);
  else
    _secderiv_sparsity = std::move(sparsity);
}

const std::optional<VariableStore::DerivSparsity> &
VariableStore::derivative_sparsity() const
{
  if (currently_assembling_nonlinear_system())
    return _deriv_sparsity_nl_sys;
  return _deriv_sparsity;
}

const std::optional<VariableStore::SecDerivSparsity> &
VariableStore::second_derivative_sparsity() const
{
  if (currently_assembling_nonlinear_system())
    return _secderiv_sparsity_nl_sys;
  return _secderiv_sparsity;
}

void
VariableStore::assign_input(const ValueMap & vals, bool allow_nonexistent)
{
  for (const auto & [name, val] : vals)
  {
    auto it = _input_variables.find(name);
    if (it == _input_variables.end())
    {
      if (allow_nonexistent)
        continue;
      else
        throw NEMLException("Trying to assign value to input variable '" + name +
                            "', but no such variable exists in model '" + _object->name() + "'.");
    }
    *it->second = val.clone();
  }
}

void
VariableStore::assign_input(const SparseVector & v, bool allow_nonexistent)
{
  for (std::size_t i = 0; i < v.layout.nvar(); i++)
  {
    auto it = _input_variables.find(v.layout.var(i));
    if (it == _input_variables.end())
    {
      if (allow_nonexistent)
        continue;
      else
        throw NEMLException("Trying to assign value to input variable '" + v.layout.var(i) +
                            "', but no such variable exists in model '" + _object->name() + "'.");
    }
    *it->second = v.tensors[i];
  }
}

void
VariableStore::assign_output(const ValueMap & vals)
{
  for (const auto & [name, val] : vals)
    output_variable(name) = val;
}

void
VariableStore::assign_output(const SparseVector & v)
{
  for (std::size_t i = 0; i < v.layout.nvar(); i++)
    output_variable(v.layout.var(i)) = v.tensors[i];
}

void
VariableStore::assign_output_derivatives(const DerivMap & derivs)
{
  for (const auto & [yname, deriv] : derivs)
  {
    auto & yvar = output_variable(yname);
    for (const auto & [xname, val] : deriv)
    {
      const auto & xvar = input_variable(xname);
      yvar.d(xvar) = val;
    }
  }
}

void
VariableStore::assign_output_derivatives(const SparseMatrix & J)
{
  const auto m = J.row_layout.nvar();
  const auto n = J.col_layout.nvar();
  for (std::size_t i = 0; i < m; i++)
  {
    auto & yvar = output_variable(J.row_layout.var(i));
    for (std::size_t j = 0; j < n; j++)
      if (J.tensors[i][j].defined())
        yvar.d(input_variable(J.col_layout.var(j))) = J.tensors[i][j];
  }
}

void
VariableStore::assign_input_stack(jit::Stack & stack)
{
  std::size_t i = 0; // stack counter
  for (auto & [xname, xvar] : input_variables())
  {
    const auto & ten = stack[i++].toTensor();
    const auto dyna_dim = ten.dim() - xvar->static_dim();
    const auto intmd_dim = xvar->intmd_dim();
    xvar->assign(Tensor(ten, dyna_dim, intmd_dim), TracerPrivilege{});
  }
  jit::drop(stack, i);
}

void
VariableStore::assign_output_stack(jit::Stack & stack, bool out, bool dout, bool d2out)
{
  neml_assert_dbg(stack.size() == 1, "Stack should have exactly one element.");
  const auto stacklist = stack.back().toTensorVector();

  std::size_t i = 0; // stack counter

  if (out)
    for (auto & [yname, yvar] : output_variables())
    {
      const auto & ten = stacklist[i++];
      const auto dyna_dim = ten.dim() - yvar->static_dim();
      const auto intmd_dim = yvar->intmd_dim();
      yvar->assign(Tensor(ten, dyna_dim, intmd_dim), TracerPrivilege{});
    }

  if (dout)
    for (auto & [yvar, xvar] : derivative_sparsity().value())
    {
      const auto & ten = stacklist[i++];
      const auto & deriv = yvar->d(*xvar);
      const auto dyna_dim = ten.dim() - deriv.intmd_dim() - deriv.base_dim();
      const auto intmd_dim = deriv.intmd_dim();
      yvar->d(*xvar) = Tensor(ten, dyna_dim, intmd_dim);
    }

  if (d2out)
    for (auto & [yvar, x1var, x2var] : second_derivative_sparsity().value())
    {
      const auto & ten = stacklist[i++];
      const auto & deriv = yvar->d2(*x1var, *x2var);
      const auto dyna_dim = ten.dim() - deriv.intmd_dim() - deriv.base_dim();
      const auto intmd_dim = deriv.intmd_dim();
      yvar->d2(*x1var, *x2var) = Tensor(ten, dyna_dim, intmd_dim);
    }

  jit::drop(stack, 1);
}

ValueMap
VariableStore::collect_input() const
{
  ValueMap vals;
  for (auto && [name, var] : input_variables())
    vals[name] = var->tensor();
  return vals;
}

SparseVector
VariableStore::collect_input(const AxisLayout & layout) const
{
  std::vector<Tensor> vals(layout.nvar());
  for (std::size_t i = 0; i < layout.nvar(); i++)
    vals[i] = input_variable(layout.var(i)).tensor();
  return SparseVector(layout, vals);
}

ValueMap
VariableStore::collect_output() const
{
  ValueMap vals;
  for (auto && [name, var] : output_variables())
    vals[name] = var->tensor();
  return vals;
}

SparseVector
VariableStore::collect_output(const AxisLayout & layout) const
{
  std::vector<Tensor> vals(layout.nvar());
  for (std::size_t i = 0; i < layout.nvar(); i++)
    vals[i] = output_variable(layout.var(i)).tensor();
  return SparseVector(layout, vals);
}

DerivMap
VariableStore::collect_output_derivatives() const
{
  DerivMap derivs;
  for (auto && [name, var] : output_variables())
    for (const auto & [deriv, xvar] : var->derivatives())
      if (deriv.defined())
        derivs[name][xvar->name()] = deriv.tensor();
  return derivs;
}

SparseMatrix
VariableStore::collect_output_derivatives(const AxisLayout & row_layout,
                                          const AxisLayout & col_layout) const
{
  const auto m = row_layout.nvar();
  const auto n = col_layout.nvar();
  std::vector<std::vector<Tensor>> derivs(m, std::vector<Tensor>(n));
  for (std::size_t i = 0; i < m; i++)
  {
    const auto & yvar = output_variable(row_layout.var(i));
    const auto & dy = yvar.derivatives();
    for (std::size_t j = 0; j < n; j++)
      for (const auto & [deriv, arg] : dy)
        if (arg->name() == col_layout.var(j) && deriv.defined())
          derivs[i][j] = deriv.tensor();
  }
  return SparseMatrix(row_layout, col_layout, derivs);
}

SecDerivMap
VariableStore::collect_output_second_derivatives() const
{
  SecDerivMap sec_derivs;
  for (auto && [name, var] : output_variables())
    for (const auto & [deriv, x1var, x2var] : var->second_derivatives())
      if (deriv.defined())
        sec_derivs[name][x1var->name()][x2var->name()] = deriv.tensor();
  return sec_derivs;
}

jit::Stack
VariableStore::collect_input_stack() const
{
  jit::Stack stack;
  for (const auto & [xname, xvar] : input_variables())
    stack.emplace_back(xvar->tensor());
  return stack;
}

jit::Stack
VariableStore::collect_output_stack(bool out, bool dout, bool d2out) const
{
  std::vector<ATensor> stacklist;

  if (out)
    for (const auto & [yname, yvar] : output_variables())
      stacklist.emplace_back(yvar->tensor());

  if (dout)
    for (const auto & [yvar, xvar] : derivative_sparsity().value())
      stacklist.emplace_back(yvar->d(*xvar).tensor());

  if (d2out)
    for (const auto & [yvar, x1var, x2var] : second_derivative_sparsity().value())
      stacklist.emplace_back(yvar->d2(*x1var, *x2var).tensor());

  return {stacklist};
}

} // namespace neml2
