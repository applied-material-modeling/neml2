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

#include "neml2/models/VariableBase.h"
#include "neml2/models/Derivative.h"
#include "neml2/misc/types.h"
#include "neml2/models/Model.h"
#include "neml2/models/DependencyResolver.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/mm.h"
#include "neml2/tensors/functions/einsum.h"
#include "neml2/tensors/jit.h"
#include "neml2/tensors/TraceableTensorShape.h"

namespace neml2
{
VariableBase::VariableBase(VariableName name_in, Model * owner, TensorShapeRef base_shape)
  : _name(std::move(name_in)),
    _owner(owner),
    _base_sizes(base_shape)
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

Size
VariableBase::dim() const
{
  return batch_dim() + base_dim();
}

Size
VariableBase::batch_dim() const
{
  return dynamic_dim() + intmd_dim();
}

Size
VariableBase::base_dim() const
{
  return Size(base_sizes().size());
}

Size
VariableBase::dynamic_dim() const
{
  return Size(dynamic_sizes().size());
}

Size
VariableBase::static_dim() const
{
  return intmd_dim() + base_dim();
}

Size
VariableBase::intmd_dim() const
{
  return Size(intmd_sizes().size());
}

TensorShapeRef
VariableBase::sizes() const
{
  return utils::add_shapes(dynamic_sizes().concrete(), intmd_sizes(), base_sizes());
}

TraceableTensorShape
VariableBase::batch_sizes() const
{
  return utils::add_traceable_shapes(dynamic_sizes(), intmd_sizes());
}

TensorShapeRef
VariableBase::base_sizes() const
{
  return _base_sizes;
}

TensorShapeRef
VariableBase::static_sizes() const
{
  return utils::add_shapes(intmd_sizes(), base_sizes());
}

TensorShapeRef
VariableBase::intmd_sizes() const
{
  return ref()->_cached_intmd_sizes;
}

Size
VariableBase::size(Size i) const
{
  i = utils::normalize_dim(i, 0, dim());
  if (i < dynamic_dim())
    return dynamic_sizes()[i].concrete();
  else if (i < batch_dim())
    return intmd_sizes()[i - dynamic_dim()];
  return _base_sizes[i - batch_dim()];
}

TraceableSize
VariableBase::batch_size(Size i) const
{
  i = utils::normalize_dim(i, 0, batch_dim());
  if (i < dynamic_dim())
    return dynamic_sizes()[i];
  return intmd_sizes()[i - dynamic_dim()];
}

Size
VariableBase::base_size(Size i) const
{
  i = utils::normalize_dim(i, 0, base_dim());
  return base_sizes()[i - batch_dim()];
}

const TraceableSize &
VariableBase::dynamic_size(Size dim) const
{
  dim = utils::normalize_dim(dim, 0, dynamic_dim());
  return dynamic_sizes()[dim];
}

Size
VariableBase::static_size(Size i) const
{
  i = utils::normalize_dim(i, 0, static_dim());
  if (i < intmd_dim())
    return intmd_sizes()[i];
  return base_sizes()[i - intmd_dim()];
}

Size
VariableBase::intmd_size(Size i) const
{
  i = utils::normalize_dim(i, 0, intmd_dim());
  return intmd_sizes()[i];
}

void
VariableBase::set_intmd_sizes(TensorShapeRef shape)
{
  neml_assert_dbg(
      owning(), "Cannot set intermediate sizes for a referencing variable '", name(), "'.");
  _cached_intmd_sizes = shape;
}

Tensor
VariableBase::zeros(const TensorOptions & options) const
{
  return Tensor::zeros({}, intmd_sizes(), base_sizes(), options);
}

bool
VariableBase::requires_grad() const
{
  return tensor().requires_grad();
}

bool
VariableBase::has_derivative(const VariableName & vname) const
{
  return std::any_of(_derivs.begin(),
                     _derivs.end(),
                     [&vname](const Derivative<1> & d) { return d.args()[0]->name() == vname; });
}

bool
VariableBase::has_derivative(const VariableName & v1name, const VariableName & v2name) const
{
  return std::any_of(_sec_derivs.begin(),
                     _sec_derivs.end(),
                     [&v1name, &v2name](const Derivative<2> & d)
                     { return d.args()[0]->name() == v1name && d.args()[1]->name() == v2name; });
}

template <std::size_t N>
static Derivative<N> &
get_derivative(std::vector<Derivative<N>> & derivs,
               const std::array<const VariableBase *, N + 1> & var_and_args)
{
  auto deriv = Derivative<N>(var_and_args);
  auto it = std::find(derivs.begin(), derivs.end(), deriv);
  if (it != derivs.end())
    return *it;
  derivs.push_back(deriv);
  return derivs.back();
}

Derivative<1> &
VariableBase::d(const VariableBase & var)
{
  return get_derivative<1>(_derivs, {this, &var});
}

const Derivative<1> &
VariableBase::d(const VariableBase & var) const
{
  for (const auto & deriv : _derivs)
    if (deriv.args()[0]->name() == var.name())
      return deriv;
  throw NEMLException("Variable '" + name().str() + "' does not have derivative with respect to '" +
                      var.name().str() + "'.");
}

Derivative<2> &
VariableBase::d(const VariableBase & var1, const VariableBase & var2)
{
  return get_derivative<2>(_sec_derivs, {this, &var1, &var2});
}

const Derivative<2> &
VariableBase::d(const VariableBase & var1, const VariableBase & var2) const
{
  for (const auto & deriv : _sec_derivs)
    if (deriv.args()[0]->name() == var1.name() && deriv.args()[1]->name() == var2.name())
      return deriv;
  throw NEMLException("Variable '" + name().str() + "' does not have derivative with respect to '" +
                      var1.name().str() + "' and '" + var2.name().str() + "'.");
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
        for (const auto & [xname, du_dx] :
             total_derivatives(dep, depu.parent, depu.parent->output_variable(depu.value)))
          assign_or_add(derivs[xname], mm(dy_du.get(), du_dx));
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
        for (const auto & [x2name, du2_dxk] :
             total_derivatives(dep, depu2.parent, depu2.parent->output_variable(depu2.value)))
          assign_or_add(sec_derivs[u1var.name()][x2name],
                        einsum("...ijq,...qk", {d2y_du1u2.get(), du2_dxk}));
    else if (dep.inbound_items().count({model, u2var.name()}))
      for (const auto & depu1 : dep.item_providers().at({model, u1var.name()}))
        for (const auto & [x1name, du1_dxj] :
             total_derivatives(dep, depu1.parent, depu1.parent->output_variable(depu1.value)))
          assign_or_add(sec_derivs[x1name][u2var.name()],
                        einsum("...ipk,...pj", {d2y_du1u2.get(), du1_dxj}));
    else
      for (const auto & depu1 : dep.item_providers().at({model, u1var.name()}))
        for (const auto & [x1name, du1_dxj] :
             total_derivatives(dep, depu1.parent, depu1.parent->output_variable(depu1.value)))
          for (const auto & depu2 : dep.item_providers().at({model, u2var.name()}))
            for (const auto & [x2name, du2_dxk] :
                 total_derivatives(dep, depu2.parent, depu2.parent->output_variable(depu2.value)))
              assign_or_add(sec_derivs[x1name][x2name],
                            einsum("...ipq,...pj,...qk", {d2y_du1u2.get(), du1_dxj, du2_dxk}));
  }

  for (auto & dy_du : yvar.derivatives())
  {
    const auto & uvar = *dy_du.args()[0];
    if (!dep.inbound_items().count({model, uvar.name()}))
      for (const auto & depu : dep.item_providers().at({model, uvar.name()}))
        for (const auto & [x1name, d2u_dx1] :
             total_second_derivatives(dep, depu.parent, depu.parent->output_variable(depu.value)))
          for (const auto & [x2name, d2u_dx1x2] : d2u_dx1)
            assign_or_add(sec_derivs[x1name][x2name],
                          einsum("...ip,...pjk", {dy_du.get(), d2u_dx1x2}));
  }

  return sec_derivs;
}
} // namespace neml2
