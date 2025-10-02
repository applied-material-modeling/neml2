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

#include "neml2/models/Variable.h"
#include "neml2/models/Model.h"
#include "neml2/models/DependencyResolver.h"
#include "neml2/models/Assembler.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/tensors.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/bmm.h"
#include "neml2/jit/utils.h"
#include "neml2/jit/TraceableTensorShape.h"
#include <ATen/ops/permute.h>

namespace neml2
{
VariableBase::VariableBase(VariableName name_in, Model * owner, TensorShapeRef lbatch_shape)
  : _name(std::move(name_in)),
    _owner(owner),
    _lbatch_sizes(lbatch_shape)
{
}

const Model &
VariableBase::owner() const
{
  neml_assert_dbg(_owner, "Owner of variable '", name(), "' has not been defined.");
  return *_owner;
}

Model &
VariableBase::owner()
{
  neml_assert_dbg(_owner, "Owner of variable '", name(), "' has not been defined.");
  return *_owner;
}

bool
VariableBase::is_state() const
{
  return _name.is_state();
}

bool
VariableBase::is_old_state() const
{
  return _name.is_old_state();
}

bool
VariableBase::is_force() const
{
  return _name.is_force();
}

bool
VariableBase::is_old_force() const
{
  return _name.is_old_force();
}

bool
VariableBase::is_residual() const
{
  return _name.is_residual();
}

bool
VariableBase::is_parameter() const
{
  return _name.is_parameter();
}

bool
VariableBase::is_solve_dependent() const
{
  return is_state() || is_residual() || is_parameter();
}

bool
VariableBase::is_dependent() const
{
  return !currently_solving_nonlinear_system() || is_solve_dependent();
}

TensorOptions
VariableBase::options() const
{
  return tensor().options();
}

Dtype
VariableBase::scalar_type() const
{
  return tensor().scalar_type();
}

Device
VariableBase::device() const
{
  return tensor().device();
}

Size
VariableBase::dim() const
{
  return tensor().dim();
}

TensorShapeRef
VariableBase::sizes() const
{
  return tensor().sizes();
}

Size
VariableBase::size(Size dim) const
{
  return tensor().size(dim);
}

bool
VariableBase::batched() const
{
  return tensor().batched();
}

Size
VariableBase::lbatch_dim() const
{
  return Size(lbatch_sizes().size());
}

Size
VariableBase::batch_dim() const
{
  return tensor().batch_dim() - lbatch_dim();
}

Size
VariableBase::base_dim() const
{
  return Size(base_sizes().size());
}

TensorShapeRef
VariableBase::lbatch_sizes() const
{
  return _lbatch_sizes;
}

TraceableTensorShape
VariableBase::batch_sizes() const
{
  return tensor().batch_sizes().slice(lbatch_dim());
}

Size
VariableBase::lbatch_size(Size dim) const
{
  auto i = dim < 0 ? lbatch_dim() + dim : dim;
  return lbatch_sizes()[i];
}

TraceableSize
VariableBase::batch_size(Size dim) const
{
  auto i = dim < 0 ? batch_dim() + dim : dim;
  return batch_sizes()[i];
}

Size
VariableBase::base_size(Size dim) const
{
  auto i = dim < 0 ? base_dim() + dim : dim;
  return base_sizes()[i];
}

Size
VariableBase::base_storage() const
{
  return utils::storage_size(base_sizes());
}

Size
VariableBase::assembly_storage() const
{
  return utils::storage_size(lbatch_sizes()) * utils::storage_size(base_sizes());
}

bool
VariableBase::requires_grad() const
{
  return tensor().requires_grad();
}

Tensor
VariableBase::make_zeros(const TraceableTensorShape & batch_shape,
                         const TensorOptions & options) const
{
  // First create a dummy tensor with batch dimensions filled with ones
  auto dummy_batch_sizes = TensorShape(batch_shape.size(), 1);
  auto B = utils::add_shapes(lbatch_sizes(), dummy_batch_sizes);
  auto tensor = Tensor::zeros(B, base_sizes(), options);
  // Then expand it to the batch shape
  if (!batch_shape.empty())
  {
    auto B = utils::add_traceable_shapes(lbatch_sizes(), batch_shape);
    tensor = tensor.batch_expand(B);
  }
  return tensor;
}

template <std::size_t N>
static Derivative<N> &
get_derivative(std::vector<Derivative<N>> & derivs,
               const VariableBase * var,
               const std::array<const VariableBase *, N> & args)
{
  auto deriv = Derivative<N>(var, args);
  auto it = std::find(derivs.begin(), derivs.end(), deriv);
  if (it != derivs.end())
    return *it;
  derivs.push_back(deriv);
  return derivs.back();
}

Derivative<1> &
VariableBase::d(const VariableBase & var)
{
  neml_assert_dbg(owning(), "Cannot assign derivative to a referencing variable '", name(), "'.");
  return get_derivative<1>(_derivs, this, {&var});
}

Derivative<2> &
VariableBase::d(const VariableBase & var1, const VariableBase & var2)
{
  neml_assert_dbg(
      owning(), "Cannot assign second derivative to a referencing variable '", name(), "'.");
  return get_derivative<2>(_sec_derivs, this, {&var1, &var2});
}

void
VariableBase::request_AD(const VariableBase & u)
{
  owner().request_AD(*this, u);
}

void
VariableBase::request_AD(const std::vector<const VariableBase *> & us)
{
  for (const auto & u : us)
  {
    neml_assert(u, "Internal error: null variable pointer.");
    owner().request_AD(*this, *u);
  }
}

void
VariableBase::request_AD(const VariableBase & u1, const VariableBase & u2)
{
  owner().request_AD(*this, u1, u2);
}

void
VariableBase::request_AD(const std::vector<const VariableBase *> & u1s,
                         const std::vector<const VariableBase *> & u2s)
{
  for (const auto & u1 : u1s)
    for (const auto & u2 : u2s)
    {
      neml_assert(u1, "Cannot request AD for a null variable.");
      neml_assert(u2, "Cannot request AD for a null variable.");
      owner().request_AD(*this, *u1, *u2);
    }
}

void
VariableBase::clear()
{
  neml_assert_dbg(owning(), "Cannot clear a referencing variable '", name(), "'.");
  clear_derivatives();
}

void
VariableBase::clear_derivatives()
{
  _derivs.clear();
  _sec_derivs.clear();
}

void
VariableBase::apply_chain_rule(const DependencyResolver<Model, VariableName> & dep)
{
  for (const auto & [model, vname] : dep.outbound_items())
    if (vname == name())
    {
      const auto & yvar = model->output_variable(vname);
      const auto derivs = total_derivatives(dep, model, yvar);
      for (const auto & [xname, dy_dx] : derivs)
        d(_owner->input_variable(xname)).set(dy_dx);
      return;
    }
}

void
VariableBase::apply_second_order_chain_rule(const DependencyResolver<Model, VariableName> & dep)
{
  for (const auto & [model, vname] : dep.outbound_items())
    if (vname == name())
    {
      const auto & yvar = model->output_variable(vname);
      const auto sec_derivs = total_second_derivatives(dep, model, yvar);
      for (const auto & [x1name, d2y_dx1] : sec_derivs)
        for (const auto & [x2name, d2y_dx1x2] : d2y_dx1)
          d(_owner->input_variable(x1name), _owner->input_variable(x2name)).set(d2y_dx1x2);
      return;
    }
}

static void
assign_or_add(Tensor & dest, const Tensor & val)
{
  if (dest.defined())
    dest = dest + val;
  else
    dest = val;
}

ValueMap
VariableBase::total_derivatives(const DependencyResolver<Model, VariableName> & dep,
                                Model * model,
                                const VariableBase & yvar) const
{
  ValueMap derivs;

  for (auto & dy_du : yvar.derivatives())
  {
    const auto & uvar = *dy_du.args()[0];
    if (dep.inbound_items().count({model, uvar.name()}))
      assign_or_add(derivs[uvar.name()], dy_du.get());
    else
      for (const auto & depu : dep.item_providers().at({model, uvar.name()}))
        for (const auto & [xname, du_dx] : total_derivatives(dep, depu.parent, uvar))
          assign_or_add(derivs[xname], bmm(dy_du.get(), du_dx));
  }

  return derivs;
}

DerivMap
VariableBase::total_second_derivatives(const DependencyResolver<Model, VariableName> & dep,
                                       Model * model,
                                       const VariableBase & yvar) const
{
  DerivMap sec_derivs;

  for (auto & d2y_du1u2 : yvar.second_derivatives())
  {
    const auto & u1var = *d2y_du1u2.args()[0];
    const auto & u2var = *d2y_du1u2.args()[1];
    if (dep.inbound_items().count({model, u1var.name()}) &&
        dep.inbound_items().count({model, u2var.name()}))
      assign_or_add(sec_derivs[u1var.name()][u2var.name()], d2y_du1u2.get());
    else if (dep.inbound_items().count({model, u1var.name()}))
      for (const auto & depu2 : dep.item_providers().at({model, u2var.name()}))
        for (const auto & [x2name, du2_dxk] : total_derivatives(dep, depu2.parent, u2var))
          assign_or_add(sec_derivs[u1var.name()][x2name],
                        Tensor(at::einsum("...ijq,...qk", {d2y_du1u2.get(), du2_dxk}),
                               utils::broadcast_batch_dim(d2y_du1u2.get(), du2_dxk)));
    else if (dep.inbound_items().count({model, u2var.name()}))
      for (const auto & depu1 : dep.item_providers().at({model, u1var.name()}))
        for (const auto & [x1name, du1_dxj] : total_derivatives(dep, depu1.parent, u1var))
          assign_or_add(sec_derivs[x1name][u2var.name()],
                        Tensor(at::einsum("...ipk,...pj", {d2y_du1u2.get(), du1_dxj}),
                               utils::broadcast_batch_dim(d2y_du1u2.get(), du1_dxj)));
    else
      for (const auto & depu1 : dep.item_providers().at({model, u1var.name()}))
        for (const auto & [x1name, du1_dxj] : total_derivatives(dep, depu1.parent, u1var))
          for (const auto & depu2 : dep.item_providers().at({model, u2var.name()}))
            for (const auto & [x2name, du2_dxk] : total_derivatives(dep, depu2.parent, u2var))
              assign_or_add(
                  sec_derivs[x1name][x2name],
                  Tensor(at::einsum("...ipq,...pj,...qk", {d2y_du1u2.get(), du1_dxj, du2_dxk}),
                         utils::broadcast_batch_dim(d2y_du1u2.get(), du1_dxj, du2_dxk)));
  }

  for (auto & dy_du : yvar.derivatives())
  {
    const auto & uvar = *dy_du.args()[0];
    if (!dep.inbound_items().count({model, uvar.name()}))
      for (const auto & depu : dep.item_providers().at({model, uvar.name()}))
        for (const auto & [x1name, d2u_dx1] : total_second_derivatives(dep, depu.parent, uvar))
          for (const auto & [x2name, d2u_dx1x2] : d2u_dx1)
            assign_or_add(sec_derivs[x1name][x2name],
                          Tensor(at::einsum("...ip,...pjk", {dy_du.get(), d2u_dx1x2}),
                                 utils::broadcast_batch_dim(dy_du.get(), d2u_dx1x2)));
  }

  return sec_derivs;
}

template <typename T>
TensorType
Variable<T>::type() const
{
  return TensorTypeEnum<T>::value;
}

template <typename T>
std::unique_ptr<VariableBase>
Variable<T>::clone(const VariableName & name, Model * owner) const
{
  if constexpr (std::is_same_v<T, Tensor>)
    return std::make_unique<Variable<Tensor>>(
        name.empty() ? this->name() : name, owner ? owner : _owner, lbatch_sizes(), base_sizes());
  else
    return std::make_unique<Variable<T>>(
        name.empty() ? this->name() : name, owner ? owner : _owner, lbatch_sizes());
}

template <typename T>
void
Variable<T>::ref(const VariableBase & var, bool ref_is_mutable)
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
  const auto * var_ptr = dynamic_cast<const Variable<T> *>(var.ref());
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
  _ref_is_mutable |= ref_is_mutable;
}

template <typename T>
void
Variable<T>::zero(const TensorOptions & options)
{
  if (owning())
    _value = T(make_zeros({}, options));
  else
  {
    neml_assert_dbg(_ref_is_mutable,
                    "Model '",
                    owner().name(),
                    "' is trying to zero a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->zero(options);
  }
}

template <typename T>
void
Variable<T>::set(const Tensor & val)
{
  if (owning())
    _value = utils::from_assembly<1>(val, {base_sizes()}, {lbatch_sizes()}, name().str());
  else
  {
    neml_assert_dbg(_ref_is_mutable,
                    "Model '",
                    owner().name(),
                    "' is trying to assign value to a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->set(val);
  }
}

template <typename T>
Tensor
Variable<T>::get() const
{
  neml_assert_dbg(_value.defined(), "Variable '", name(), "' has undefined value.");
  return utils::to_assembly<1>(_value, {base_sizes()}, {lbatch_sizes()}, name().str());
}

template <typename T>
Tensor
Variable<T>::tensor() const
{
  if (owning())
    return _value;

  return ref()->tensor();
}

template <typename T>
void
Variable<T>::requires_grad_(bool req)
{
  if (owning())
    _value.requires_grad_(req);
  else
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->requires_grad_(req);
}

template <typename T>
void
Variable<T>::operator=(const Tensor & val)
{
  if (owning())
  {
    neml_assert_dbg(val.defined(), "Variable '", name(), "' is being assigned an undefined value.");
    neml_assert_dbg(val.base_sizes() == base_sizes(),
                    "Variable '",
                    name(),
                    "' is being assigned a value with incompatible base sizes. Expected: ",
                    base_sizes(),
                    ", Got: ",
                    val.base_sizes());
    neml_assert_dbg(val.batch_dim() >= lbatch_dim() &&
                        val.batch_sizes().slice(0, lbatch_dim()) == lbatch_sizes(),
                    "Variable '",
                    name(),
                    "' is being assigned a value with incompatible left-batch shape. Expected "
                    "left-batch shape: ",
                    lbatch_sizes(),
                    ", Got batch shape: ",
                    val.batch_sizes());
    _value = T(val);
  }
  else
  {
    neml_assert_dbg(_ref_is_mutable,
                    "Model '",
                    owner().name(),
                    "' is trying to assign value to a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    *const_cast<VariableBase *>(ref()) = val;
  }
}

template <typename T>
void
Variable<T>::assign(const ATensor & val, RawAssignment key)
{
  if (owning())
  {
    if constexpr (std::is_same_v<T, Tensor>)
      _value = T(val, val.dim() - base_dim());
    else
      _value = T(val);
  }
  else
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->assign(val, key);
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
    neml_assert_dbg(_ref_is_mutable,
                    "Model '",
                    owner().name(),
                    "' is trying to clear a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->clear();
  }
}

#define INSTANTIATE_VARIABLE(T) template class Variable<T>
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_VARIABLE);

template <std::size_t N>
Derivative<N>::Derivative(const VariableBase * var,
                          const std::array<const VariableBase *, N> & args)
  : _var(var),
    _args(args)
#ifndef NDEBUG
    ,
    _debug_name(N == 1
                    ? std::string("d(") + var->name().str() + ")/d(" + args[0]->name().str() + ")"
                    : std::string("d2(") + var->name().str() + ")/d(" + args[0]->name().str() +
                          ")d(" + args[1]->name().str() + ")")
#endif
{
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator=(const Tensor & val)
{
#ifndef NDEBUG
  auto lbatch_sizes = N == 1 ? utils::add_shapes(_var->lbatch_sizes(), _args[0]->lbatch_sizes())
                             : utils::add_shapes(_var->lbatch_sizes(),
                                                 _args[0]->lbatch_sizes(),
                                                 _args[1]->lbatch_sizes());
  auto base_sizes =
      N == 1
          ? utils::add_shapes(_var->base_sizes(), _args[0]->base_sizes())
          : utils::add_shapes(_var->base_sizes(), _args[0]->base_sizes(), _args[1]->base_sizes());
  auto lbatch_dim = Size(lbatch_sizes.size());
  neml_assert_dbg(val.batch_dim() >= lbatch_dim,
                  "The assigned derivative for variable '",
                  _debug_name,
                  "' has insufficient batch dimensions. Expected at least ",
                  lbatch_dim,
                  ", got ",
                  val.batch_dim(),
                  ".");
  neml_assert_dbg(val.batch_sizes().slice(0, lbatch_dim) == lbatch_sizes,
                  "The assigned derivative for variable '",
                  _debug_name,
                  "' has incorrect left-batch shape. Expected ",
                  lbatch_sizes,
                  ", got ",
                  val.batch_sizes().slice(0, lbatch_dim));
  neml_assert_dbg(val.base_sizes() == base_sizes,
                  "The assigned derivative for variable '",
                  _debug_name,
                  "' has incorrect base shape. Expected ",
                  base_sizes,
                  ", got ",
                  val.base_sizes());
#endif
  assign_or_add(_deriv, val);

  // Invalidate the assembly cache
  _deriv_assembly = Tensor();

  return *this;
}

template <std::size_t N>
Derivative<N> &
Derivative<N>::operator=(const VariableBase & val)
{
  return Derivative<N>::operator=(val.tensor());
}

template <std::size_t N>
const Tensor &
Derivative<N>::get() const
{
  if (_deriv_assembly.defined())
    return _deriv_assembly;

  if constexpr (N == 1)
    _deriv_assembly = utils::to_assembly<2>(_deriv,
                                            {_var->base_sizes(), _args[0]->base_sizes()},
                                            {_var->lbatch_sizes(), _args[0]->lbatch_sizes()});
  else
    _deriv_assembly = utils::to_assembly<3>(
        _deriv,
        {_var->base_sizes(), _args[0]->base_sizes(), _args[1]->base_sizes()},
        {_var->lbatch_sizes(), _args[0]->lbatch_sizes(), _args[1]->lbatch_sizes()});
  return _deriv_assembly;
}

template <std::size_t N>
void
Derivative<N>::set(const Tensor & val)
{
  _deriv_assembly = val;

  if constexpr (N == 1)
    _deriv = utils::from_assembly<2>(val,
                                     {_var->base_sizes(), _args[0]->base_sizes()},
                                     {_var->lbatch_sizes(), _args[0]->lbatch_sizes()});
  else
    _deriv = utils::from_assembly<3>(
        val,
        {_var->base_sizes(), _args[0]->base_sizes(), _args[1]->base_sizes()},
        {_var->lbatch_sizes(), _args[0]->lbatch_sizes(), _args[1]->lbatch_sizes()});
}

template class Derivative<1>;
template class Derivative<2>;

}
