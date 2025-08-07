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
#include "neml2/misc/assertions.h"
#include "neml2/models/map_types.h"
#include "neml2/models/Variable.h"
#include "neml2/base/LabeledAxis.h"
#include "neml2/solvers/NonlinearSystem.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/sum.h"

namespace neml2
{
VariableStore::VariableStore(Model * object)
  : _object(object),
    _input_axis(declare_axis("input")),
    _output_axis(declare_axis("output")),
    _options(default_tensor_options())
{
}

LabeledAxis &
VariableStore::declare_axis(const std::string & name)
{
  neml_assert(!_axes.count(name),
              "Trying to declare an axis named ",
              name,
              ", but an axis with the same name already exists.");

  auto axis = std::make_unique<LabeledAxis>();
  auto [it, success] = _axes.emplace(name, std::move(axis));
  return *it->second;
}

void
VariableStore::setup_layout()
{
  input_axis().setup_layout();
  output_axis().setup_layout();
}

template <typename T>
const Variable<T> &
VariableStore::declare_input_variable(const char * name,
                                      TensorShapeRef lbatch_shape,
                                      bool allow_duplicate)
{
  if (_object->input_options().contains(name))
    return declare_input_variable<T>(
        _object->input_options().get<VariableName>(name), lbatch_shape, allow_duplicate);

  return declare_input_variable<T>(VariableName(name), lbatch_shape, allow_duplicate);
}

template <typename T>
const Variable<T> &
VariableStore::declare_input_variable(const VariableName & name,
                                      TensorShapeRef lbatch_shape,
                                      bool allow_duplicate)
{
  const auto lbatch_sz = utils::storage_size(lbatch_shape);
  const auto base_sz = T::const_base_storage;
  const auto sz = lbatch_sz * base_sz;

  if (!allow_duplicate || (allow_duplicate && !_input_axis.has_variable(name)))
    _input_axis.add_variable(name, sz);
  return *create_variable<T>(_input_variables, name, lbatch_shape, allow_duplicate);
}
#define INSTANTIATE_DECLARE_INPUT_VARIABLE(T)                                                      \
  template const Variable<T> & VariableStore::declare_input_variable<T>(                           \
      const char *, TensorShapeRef, bool);                                                         \
  template const Variable<T> & VariableStore::declare_input_variable<T>(                           \
      const VariableName &, TensorShapeRef, bool)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_DECLARE_INPUT_VARIABLE);

template <typename T>
Variable<T> &
VariableStore::declare_output_variable(const char * name, TensorShapeRef lbatch_shape)
{
  if (_object->input_options().contains(name))
    return declare_output_variable<T>(_object->input_options().get<VariableName>(name),
                                      lbatch_shape);

  return declare_output_variable<T>(VariableName(name), lbatch_shape);
}

template <typename T>
Variable<T> &
VariableStore::declare_output_variable(const VariableName & name, TensorShapeRef lbatch_shape)
{
  const auto lbatch_sz = utils::storage_size(lbatch_shape);
  const auto base_sz = T::const_base_storage;
  const auto sz = lbatch_sz * base_sz;

  _output_axis.add_variable(name, sz);
  return *create_variable<T>(_output_variables, name, lbatch_shape);
}
#define INSTANTIATE_DECLARE_OUTPUT_VARIABLE(T)                                                     \
  template Variable<T> & VariableStore::declare_output_variable<T>(const char *, TensorShapeRef);  \
  template Variable<T> & VariableStore::declare_output_variable<T>(const VariableName &,           \
                                                                   TensorShapeRef)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_DECLARE_OUTPUT_VARIABLE);

const VariableBase *
VariableStore::clone_input_variable(const VariableBase & var, const VariableName & new_name)
{
  neml_assert(&var.owner() != _object, "Trying to clone a variable from the same model.");

  const auto var_name = new_name.empty() ? var.name() : new_name;
  neml_assert(
      !_input_variables.count(var_name), "Input variable '", var_name.str(), "' already exists.");
  auto var_clone = var.clone(var_name, _object);

  _input_axis.add_variable(var_name, var_clone->assembly_storage());
  auto [it, success] = _input_variables.emplace(var_name, std::move(var_clone));
  return it->second.get();
}

VariableBase *
VariableStore::clone_output_variable(const VariableBase & var, const VariableName & new_name)
{
  neml_assert(&var.owner() != _object, "Trying to clone a variable from the same model.");

  const auto var_name = new_name.empty() ? var.name() : new_name;
  neml_assert(
      !_output_variables.count(var_name), "Output variable '", var_name, "' already exists.");
  auto var_clone = var.clone(var_name, _object);

  _output_axis.add_variable(var_name, var_clone->assembly_storage());
  auto [it, success] = _output_variables.emplace(var_name, std::move(var_clone));
  return it->second.get();
}

template <typename T>
Variable<T> *
VariableStore::create_variable(VariableStorage & variables,
                               const VariableName & name,
                               TensorShapeRef lbatch_shape,
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
    var = std::make_unique<Variable<T>>(name, _object, lbatch_shape);
    auto [it, success] = variables.emplace(name, std::move(var));
    var_base_ptr = it->second.get();
  }

  // Cast it to the concrete type
  auto var_ptr = dynamic_cast<Variable<T> *>(var_base_ptr);
  if (!var_ptr)
    throw NEMLException("Internal error: Failed to cast variable '" + name.str() +
                        "' to its concrete type.");

  return var_ptr;
}
#define INSTANTIATE_CREATE_VARIABLE(T)                                                             \
  template Variable<T> * VariableStore::create_variable<T>(                                        \
      VariableStorage &, const VariableName &, TensorShapeRef, bool)
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
    for (const auto & dy_dx : yvar->derivatives())
      sparsity.emplace_back(yvar.get(), dy_dx.args()[0]);

  if (currently_solving_nonlinear_system())
    _deriv_sparsity_nl_sys = std::move(sparsity);
  else
    _deriv_sparsity = std::move(sparsity);
}

void
VariableStore::cache_second_derivative_sparsity()
{
  std::vector<std::tuple<VariableBase *, const VariableBase *, const VariableBase *>> sparsity;
  for (auto && [yname, yvar] : output_variables())
    for (const auto & d2y_dx1dx2 : yvar->second_derivatives())
      sparsity.emplace_back(yvar.get(), d2y_dx1dx2.args()[0], d2y_dx1dx2.args()[1]);

  if (currently_solving_nonlinear_system())
    _secderiv_sparsity_nl_sys = std::move(sparsity);
  else
    _secderiv_sparsity = std::move(sparsity);
}

const std::optional<VariableStore::DerivSparsity> &
VariableStore::derivative_sparsity() const
{
  if (currently_solving_nonlinear_system())
    return _deriv_sparsity_nl_sys;
  return _deriv_sparsity;
}

const std::optional<VariableStore::SecDerivSparsity> &
VariableStore::second_derivative_sparsity() const
{
  if (currently_solving_nonlinear_system())
    return _secderiv_sparsity_nl_sys;
  return _secderiv_sparsity;
}

void
VariableStore::assign_input(const ValueMap & vals, bool assembly)
{
  for (const auto & [name, val] : vals)
    if (assembly)
      input_variable(name).set(val.clone());
    else
      input_variable(name) = val.clone();
}

void
VariableStore::assign_output(const ValueMap & vals, bool assembly)
{
  for (const auto & [name, val] : vals)
    if (assembly)
      output_variable(name).set(val);
    else
      output_variable(name) = val;
}

void
VariableStore::assign_output_derivatives(const DerivMap & derivs, bool assembly)
{
  for (const auto & [yname, deriv] : derivs)
  {
    auto & yvar = output_variable(yname);
    for (const auto & [xname, val] : deriv)
    {
      const auto & xvar = input_variable(xname);
      auto & dy_dx = yvar.d(xvar);
      if (assembly)
        dy_dx.set(val);
      else
        dy_dx = val;
    }
  }
}

void
VariableStore::assign_input_stack(jit::Stack & stack)
{
  std::size_t i = 0; // stack counter
  for (auto & [xname, xvar] : input_variables())
  {
    const auto & ten = stack[i++].toTensor();
    xvar->assign(Tensor(ten, ten.dim() - xvar->base_dim()), VariableBase::RawAssignment{});
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
      yvar->assign(Tensor(ten, ten.dim() - yvar->base_dim()), VariableBase::RawAssignment{});
    }

  if (dout)
    for (auto & [yvar, xvar] : derivative_sparsity().value())
    {
      const auto & ten = stacklist[i++];
      yvar->d(*xvar) = Tensor(ten, ten.dim() - yvar->base_dim() - xvar->base_dim());
    }

  if (d2out)
    for (auto & [yvar, x1var, x2var] : second_derivative_sparsity().value())
    {
      const auto & ten = stacklist[i++];
      yvar->d(*x1var, *x2var) =
          Tensor(ten, ten.dim() - yvar->base_dim() - x1var->base_dim() - x2var->base_dim());
    }

  jit::drop(stack, 1);
}

ValueMap
VariableStore::collect_input(bool assembly) const
{
  ValueMap vals;
  for (auto && [name, var] : input_variables())
    if (assembly)
      vals[name] = var->get();
    else
      vals[name] = var->tensor();
  return vals;
}

ValueMap
VariableStore::collect_output(bool assembly) const
{
  ValueMap vals;
  for (auto && [name, var] : output_variables())
    if (assembly)
      vals[name] = var->get();
    else
      vals[name] = var->tensor();
  return vals;
}

DerivMap
VariableStore::collect_output_derivatives(bool assembly) const
{
  DerivMap derivs;
  for (auto && [name, var] : output_variables())
    for (auto & deriv : var->derivatives())
      if (assembly)
        derivs[name][deriv.args()[0]->name()] = deriv.get();
      else
        derivs[name][deriv.args()[0]->name()] = deriv.tensor();
  return derivs;
}

SecDerivMap
VariableStore::collect_output_second_derivatives(bool assembly) const
{
  SecDerivMap sec_derivs;
  for (auto && [name, var] : output_variables())
    for (auto & deriv : var->second_derivatives())
      if (assembly)
        sec_derivs[name][deriv.args()[0]->name()][deriv.args()[1]->name()] = deriv.get();
      else
        sec_derivs[name][deriv.args()[0]->name()][deriv.args()[1]->name()] = deriv.tensor();
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
      if (xvar->is_dependent())
        stacklist.emplace_back(yvar->d(*xvar).tensor());

  if (d2out)
    for (const auto & [yvar, x1var, x2var] : second_derivative_sparsity().value())
      if (x1var->is_dependent() && x2var->is_dependent())
        stacklist.emplace_back(yvar->d(*x1var, *x2var).tensor());

  return {stacklist};
}

} // namespace neml2
