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

#include <ATen/ExpandUtils.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include "neml2/models/Variable.h"
#include "neml2/tensors/jit.h"
#include "neml2/tensors/Derivative.h"
#include "neml2/misc/assertions.h"
#include "neml2/models/Model.h"

namespace neml2
{
template <typename T>
TensorType
Variable<T>::type() const
{
  return TensorTypeEnum<T>::value;
}

template <typename T>
bool
Variable<T>::defined() const
{
  return operator()().defined();
}

template <typename T>
TensorOptions
Variable<T>::options() const
{
  return operator()().options();
}

template <typename T>
Dtype
Variable<T>::scalar_type() const
{
  return operator()().scalar_type();
}

template <typename T>
Device
Variable<T>::device() const
{
  return operator()().device();
}

template <typename T>
const TraceableTensorShape &
Variable<T>::dynamic_sizes() const
{
  return operator()().dynamic_sizes();
}

template <typename T>
std::unique_ptr<VariableBase>
Variable<T>::clone(const VariableName & name, Model * owner) const
{
  return std::make_unique<Variable<T>>(name.empty() ? this->name() : name, owner ? owner : _owner);
}

template <typename T>
void
Variable<T>::ref(VariableBase & var)
{
  neml_assert(!_ref || ref() == var.ref(),
              "Variable '",
              name(),
              "' cannot reference another variable '",
              var.name(),
              "' after it has been assigned a reference. \nThe "
              "existing reference '",
              ref()->name(),
              "' was declared by model '",
              ref()->owner().name(),
              "'. \nThe new reference is declared by model '",
              var.owner().name(),
              "'.");
  neml_assert(&var != this, "Variable '", name(), "' cannot reference itself.");
  neml_assert(var.ref() != this,
              "Variable '",
              name(),
              "' cannot reference a variable that is referencing itself.");
  auto * var_ptr = dynamic_cast<Variable<T> *>(&var);
  neml_assert(var_ptr,
              "Variable ",
              name(),
              " of type ",
              type(),
              " failed to reference another variable named ",
              var.name(),
              " of type ",
              var.type(),
              ": Dynamic cast failure.");
  _ref = var_ptr;
  ref()->set_mutable(ref()->is_mutable() || _mutable);
}

template <typename T>
void
Variable<T>::zero(const TensorOptions & options)
{
  if (owning())
    _value = VariableBase::zeros(options);
  else
  {
    neml_assert_dbg(ref()->is_mutable(),
                    "Model '",
                    owner().name(),
                    "' is trying to zero a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    ref()->zero(options);
  }
}

template <typename T>
Tensor
Variable<T>::tensor() const
{
  if (!owning())
    return ref()->tensor();

  if (!jit::tracer::isTracing())
    neml_assert_dbg(
        _value.defined() && _value.isfinite().all().template item<bool>(),
        "Variable '",
        name(),
        "' has undefined or non-finite value. Below is some additional debugging information:\n",
        "  owner: ",
        owner().name(),
        "\n  owning: ",
        owning() ? "true" : "false",
        "\n  defined: ",
        _value.defined() ? "true" : "false",
        "\n  isfinite: ",
        _value.defined() && _value.isfinite().all().template item<bool>() ? "true" : "false");
  return _value;
}

template <typename T>
void
Variable<T>::requires_grad_(bool req)
{
  if (owning())
    _value.requires_grad_(req);
  else
    ref()->requires_grad_(req);
}

template <typename T>
void
Variable<T>::assign(const Tensor & val, [[maybe_unused]] std::optional<TracerPrivilege> key)
{
  if (owning())
  {
    neml_assert_dbg(val.base_sizes() == base_sizes(),
                    "Variable '",
                    name(),
                    "' is being assigned a value with incompatible base sizes. Expected: ",
                    base_sizes(),
                    ", Got: ",
                    val.base_sizes());
    neml_assert_dbg(val.defined() && val.isfinite().all().template item<bool>(),
                    "Variable '",
                    name(),
                    "' is assigned with undefined or non-finite value. Below is some additional "
                    "debugging information:\n",
                    "  owner: ",
                    owner().name(),
                    "\n  owning: ",
                    owning() ? "true" : "false",
                    "\n  defined: ",
                    val.defined() ? "true" : "false",
                    "\n  isfinite: ",
                    val.defined() && val.isfinite().all().template item<bool>() ? "true" : "false");
    _value = T(val);
    _cached_intmd_sizes = val.intmd_sizes();
  }
  else
  {
    neml_assert_dbg(ref()->is_mutable() || key.has_value(),
                    "Model '",
                    owner().name(),
                    "' is trying to assign value to a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    ref()->assign(val);
  }
}

template <typename T>
void
Variable<T>::operator=(const Tensor & val)
{
  assign(val);
}

template <typename T>
void
Variable<T>::clear()
{
  if (owning())
  {
    VariableBase::clear();
    _value = T();
  }
  else
  {
    neml_assert_dbg(ref()->is_mutable(),
                    "Model '",
                    owner().name(),
                    "' is trying to clear a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    ref()->clear();
  }
}

#define INSTANTIATE_VARIABLE(T) template class Variable<T>
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_VARIABLE);
} // namespace neml2
