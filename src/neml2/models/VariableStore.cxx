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
                                      TensorShapeRef list_shape,
                                      bool allow_duplicate)
{
  if (_object->input_options().contains(name))
    return declare_input_variable<T>(
        _object->input_options().get<VariableName>(name), list_shape, allow_duplicate);

  return declare_input_variable<T>(VariableName(name), list_shape, allow_duplicate);
}

template <typename T>
const Variable<T> &
VariableStore::declare_input_variable(const VariableName & name,
                                      TensorShapeRef list_shape,
                                      bool allow_duplicate)
{
  const auto list_sz = utils::storage_size(list_shape);
  const auto base_sz = T::const_base_storage;
  const auto sz = list_sz * base_sz;

  if (!allow_duplicate || (allow_duplicate && !_input_axis.has_variable(name)))
    _input_axis.add_variable(name, sz);
  return *create_variable<T>(_input_variables, name, list_shape, allow_duplicate);
}
#define INSTANTIATE_DECLARE_INPUT_VARIABLE(T)                                                      \
  template const Variable<T> & VariableStore::declare_input_variable<T>(                           \
      const char *, TensorShapeRef, bool);                                                         \
  template const Variable<T> & VariableStore::declare_input_variable<T>(                           \
      const VariableName &, TensorShapeRef, bool)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_DECLARE_INPUT_VARIABLE);

template <typename T>
Variable<T> &
VariableStore::declare_output_variable(const char * name, TensorShapeRef list_shape)
{
  if (_object->input_options().contains(name))
    return declare_output_variable<T>(_object->input_options().get<VariableName>(name), list_shape);

  return declare_output_variable<T>(VariableName(name), list_shape);
}

template <typename T>
Variable<T> &
VariableStore::declare_output_variable(const VariableName & name, TensorShapeRef list_shape)
{
  const auto list_sz = utils::storage_size(list_shape);
  const auto base_sz = T::const_base_storage;
  const auto sz = list_sz * base_sz;

  _output_axis.add_variable(name, sz);
  return *create_variable<T>(_output_variables, name, list_shape);
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
                               TensorShapeRef list_shape,
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
    var = std::make_unique<Variable<T>>(name, _object, list_shape);
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
VariableStore::zero_input()
{
  for (auto && [name, var] : input_variables())
    if (var->owning())
      var->zero(_options);
}

void
VariableStore::zero_output()
{
  for (auto && [name, var] : output_variables())
    if (var->owning())
      var->zero(_options);
}

void
VariableStore::assign_input(const ValueMap & vals)
{
  for (const auto & [name, val] : vals)
    if (input_axis().has_variable(name))
      input_variable(name).set(val.clone());
}

void
VariableStore::assign_input(ValueMap && vals)
{
  for (const auto & [name, val] : std::move(vals))
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

void
VariableStore::assign_input_stack(jit::Stack & stack)
{
  const auto & vars = input_axis().variable_names();
  neml_assert_dbg(stack.size() >= vars.size(),
                  "Number of input variables in the stack (",
                  stack.size(),
                  ") is smaller than the number of input variables in the model (",
                  vars.size(),
                  ").");

  // Last n tensors in the stack are the input variables
  for (std::size_t i = 0; i < vars.size(); i++)
    input_variable(vars[i]).set(stack[stack.size() - vars.size() + i].toTensor(), /*force=*/true);

  // Drop the input variables from the stack
  jit::drop(stack, vars.size());
}

void
VariableStore::assign_output_stack(jit::Stack & stack, bool out, bool dout, bool d2out)
{
  neml_assert_dbg(out || dout || d2out,
                  "At least one of the output/derivative flags must be true.");

  neml_assert_dbg(!stack.empty(), "Empty output stack.");
  const auto stacklist = stack.back().toTensorVector();

  // With our protocol, the last tensor in the list is the sparsity tensor
  const auto sparsity_tensor = stacklist.back().contiguous();
  neml_assert_dbg(at::sum(sparsity_tensor).item<Size>() == Size(stacklist.size()) - 1,
                  "Sparsity tensor has incorrect size. Got ",
                  at::sum(sparsity_tensor).item<Size>(),
                  " expected ",
                  Size(stacklist.size()) - 1);
  const std::vector<Size> sparsity(sparsity_tensor.data_ptr<Size>(),
                                   sparsity_tensor.data_ptr<Size>() + sparsity_tensor.size(0));

  const auto & yvars = output_axis().variable_names();
  const auto & xvars = input_axis().variable_names();

  std::size_t sti = 0; // stack counter
  std::size_t spi = 0; // sparsity counter

  if (out)
  {
    for (const auto & yvar : yvars)
    {
      neml_assert(sparsity[spi++], "Corrupted sparsity tensor.");
      output_variable(yvar).set(stacklist[sti++], /*force=*/true);
    }
  }

  if (dout)
  {
    for (const auto & yvar : yvars)
    {
      auto & derivs = output_variable(yvar).derivatives();
      for (const auto & xvar : xvars)
      {
        if (sparsity[spi++])
        {
          const auto & val = stacklist[sti++];
          neml_assert_dbg(val.dim() >= 2,
                          "Derivative tensor d(",
                          yvar,
                          ")/d(",
                          xvar,
                          ") must have at least 2 dimensions. Got ",
                          val.dim(),
                          ".");
          derivs[xvar] = Tensor(val, val.dim() - 2);
        }
      }
    }
  }

  if (d2out)
  {
    for (const auto & yvar : yvars)
    {
      auto & derivs = output_variable(yvar).second_derivatives();
      for (const auto & x1var : xvars)
        for (const auto & x2var : xvars)
        {
          if (sparsity[spi++])
          {
            const auto & val = stacklist[sti++];
            neml_assert_dbg(val.dim() >= 3,
                            "Second derivative tensor d2(",
                            yvar,
                            ")/d(",
                            x1var,
                            ")d(",
                            x2var,
                            ") must have at least 3 dimensions. Got ",
                            val.dim(),
                            ".");
            derivs[x1var][x2var] = Tensor(val, val.dim() - 3);
          }
        }
    }
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

ValueMap
VariableStore::collect_output() const
{
  ValueMap vals;
  for (auto && [name, var] : output_variables())
    vals[name] = var->tensor();
  return vals;
}

DerivMap
VariableStore::collect_output_derivatives() const
{
  DerivMap derivs;
  for (auto && [name, var] : output_variables())
    derivs[name] = var->derivatives();
  return derivs;
}

SecDerivMap
VariableStore::collect_output_second_derivatives() const
{
  SecDerivMap sec_derivs;
  for (auto && [name, var] : output_variables())
    sec_derivs[name] = var->second_derivatives();
  return sec_derivs;
}

jit::Stack
VariableStore::collect_input_stack() const
{
  jit::Stack stack;
  const auto & vars = input_axis().variable_names();
  stack.reserve(vars.size());
  for (const auto & name : vars)
    stack.emplace_back(input_variable(name).tensor());
  return stack;
}

jit::Stack
VariableStore::collect_output_stack(bool out, bool dout, bool d2out) const
{
  neml_assert_dbg(out || dout || d2out,
                  "At least one of the output/derivative flags must be true.");

  const auto & yvars = output_axis().variable_names();
  const auto & xvars = input_axis().variable_names();

  std::vector<ATensor> stacklist;
  std::vector<Size> sparsity;

  if (out)
  {
    sparsity.insert(sparsity.end(), yvars.size(), 1);
    for (const auto & yvar : yvars)
      stacklist.push_back(output_variable(yvar).tensor());
  }

  if (dout)
  {
    for (const auto & yvar : yvars)
    {
      const auto & derivs = output_variable(yvar).derivatives();
      for (const auto & xvar : xvars)
      {
        const auto & deriv = derivs.find(xvar);
        sparsity.push_back(deriv == derivs.end() || !input_variable(xvar).is_dependent() ? 0 : 1);
        if (sparsity.back())
          stacklist.push_back(deriv->second);
      }
    }
  }

  if (d2out)
  {
    for (const auto & yvar : yvars)
    {
      const auto & derivs = output_variable(yvar).second_derivatives();
      for (const auto & x1var : xvars)
      {
        const auto & x1derivs = derivs.find(x1var);
        if (x1derivs != derivs.end() && input_variable(x1var).is_dependent())
          for (const auto & x2var : xvars)
          {
            const auto & x1x2deriv = x1derivs->second.find(x2var);
            sparsity.push_back(
                x1x2deriv == x1derivs->second.end() || !input_variable(x2var).is_dependent() ? 0
                                                                                             : 1);
            if (sparsity.back())
              stacklist.push_back(x1x2deriv->second);
          }
        else
          sparsity.insert(sparsity.end(), xvars.size(), 0);
      }
    }
  }

  const auto sparsity_tensor = Tensor::create(sparsity, kInt64);
  const auto nnz = base_sum(sparsity_tensor).item<Size>();
  neml_assert_dbg(nnz == Size(stacklist.size()),
                  "Corrupted sparsity tensor. Got ",
                  nnz,
                  " non-zero entries, expected ",
                  Size(stacklist.size()));
  stacklist.push_back(sparsity_tensor);

  return {stacklist};
}

} // namespace neml2
