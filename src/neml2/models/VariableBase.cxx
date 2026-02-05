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
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/tensors/Derivative.h"
#include "neml2/misc/types.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/chain_rule.h"
#include "neml2/tensors/jit.h"
#include "neml2/tensors/TraceableTensorShape.h"
#include "neml2/models/DependencyResolver.h"

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
  return !currently_assembling_nonlinear_system() || is_solve_dependent();
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

bool
VariableBase::is_mutable() const
{
  if (!owning())
    return ref()->is_mutable();
  return _mutable;
}

void
VariableBase::set_mutable(bool m)
{
  if (!owning())
    ref()->set_mutable(m);
  _mutable = m;
}

Tensor
VariableBase::zeros(const TensorOptions & options) const
{
  return Tensor::zeros({}, {}, base_sizes(), options);
}

bool
VariableBase::requires_grad() const
{
  return tensor().requires_grad();
}

bool
VariableBase::has_derivative(const VariableName & vname) const
{
  for (const auto & [deriv, arg] : _derivs)
    if (arg->name() == vname && deriv.defined())
      return true;
  return false;
}

bool
VariableBase::has_derivative(const VariableName & v1name, const VariableName & v2name) const
{
  for (const auto & [deriv, arg1, arg2] : _sec_derivs)
    if (arg1->name() == v1name && arg2->name() == v2name && deriv.defined())
      return true;
  return false;
}

static Derivative<1> &
get_deriv(VariableBase::DerivContainer & derivs,
          const VariableBase & var,
          const VariableBase & arg,
          std::size_t deriv_intrsc_intmd_dim,
          std::size_t var_intrsc_intmd_dim,
          std::size_t arg_intrsc_intmd_dim)
{
  // Check if derivative already exists
  for (auto & [deriv, a] : derivs)
    if (a->name() == arg.name())
      return deriv;

  // Make a new derivative
  const auto intmd_sizes = std::array{var.intmd_sizes(), arg.intmd_sizes()};
  const auto base_sizes = std::array{var.base_sizes(), arg.base_sizes()};
  auto & deriv = derivs.emplace_back(Derivative<1>(deriv_intrsc_intmd_dim,
                                                   {var_intrsc_intmd_dim, arg_intrsc_intmd_dim},
                                                   intmd_sizes,
                                                   base_sizes,
                                                   var.name().str(),
                                                   {arg.name().str()}),
                                     &arg);
  return std::get<0>(deriv);
}

Derivative<1> &
VariableBase::d(const VariableBase & arg,
                std::size_t deriv_intrsc_intmd_dim,
                std::size_t var_intrsc_intmd_dim,
                std::size_t arg_intrsc_intmd_dim)
{
  return get_deriv(
      _derivs, *this, arg, deriv_intrsc_intmd_dim, var_intrsc_intmd_dim, arg_intrsc_intmd_dim);
}

const Derivative<1> &
VariableBase::d(const VariableBase & arg) const
{
  for (const auto & [deriv, a] : _derivs)
    if (a->name() == arg.name())
      return deriv;
  throw NEMLException("Variable '" + name().str() + "' does not have derivative with respect to '" +
                      arg.name().str() + "'.");
}

static Derivative<2> &
get_secderiv(VariableBase::SecDerivContainer & secderivs,
             const VariableBase & var,
             const VariableBase & arg1,
             const VariableBase & arg2,
             std::size_t deriv_intrsc_intmd_dim,
             std::size_t var_intrsc_intmd_dim,
             std::size_t arg1_intrsc_intmd_dim,
             std::size_t arg2_intrsc_intmd_dim)
{
  // Check if derivative already exists
  for (auto & [deriv, a1, a2] : secderivs)
    if (a1->name() == arg1.name() && a2->name() == arg2.name())
      return deriv;

  // Make a new derivative
  const auto intmd_sizes = std::array{var.intmd_sizes(), arg1.intmd_sizes(), arg2.intmd_sizes()};
  const auto base_sizes = std::array{var.base_sizes(), arg1.base_sizes(), arg2.base_sizes()};
  auto & deriv = secderivs.emplace_back(
      Derivative<2>(deriv_intrsc_intmd_dim,
                    {var_intrsc_intmd_dim, arg1_intrsc_intmd_dim, arg2_intrsc_intmd_dim},
                    intmd_sizes,
                    base_sizes,
                    var.name().str(),
                    {arg1.name().str(), arg2.name().str()}),
      &arg1,
      &arg2);
  return std::get<0>(deriv);
}

Derivative<2> &
VariableBase::d2(const VariableBase & arg1,
                 const VariableBase & arg2,
                 std::size_t deriv_intrsc_intmd_dim,
                 std::size_t var_intrsc_intmd_dim,
                 std::size_t arg1_intrsc_intmd_dim,
                 std::size_t arg2_intrsc_intmd_dim)
{
  return get_secderiv(_sec_derivs,
                      *this,
                      arg1,
                      arg2,
                      deriv_intrsc_intmd_dim,
                      var_intrsc_intmd_dim,
                      arg1_intrsc_intmd_dim,
                      arg2_intrsc_intmd_dim);
}

const Derivative<2> &
VariableBase::d2(const VariableBase & arg1, const VariableBase & arg2) const
{
  for (const auto & [deriv, a1, a2] : _sec_derivs)
    if (a1->name() == arg1.name() && a2->name() == arg2.name())
      return deriv;
  throw NEMLException("Variable '" + name().str() + "' does not have derivative with respect to '" +
                      arg1.name().str() + "' and '" + arg2.name().str() + "'.");
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
  for (auto & [deriv, arg] : _derivs)
    deriv.clear();
  for (auto & [deriv, arg1, arg2] : _sec_derivs)
    deriv.clear();
}

bool
VariableBase::is_leaf(const DependencyResolver<Model, VariableName> & deps) const
{
  return deps.inbound_items().count({_owner, name()});
}

const VariableBase &
VariableBase::provider(const DependencyResolver<Model, VariableName> & deps) const
{
  if (owning())
    return *this;

#ifndef NDEBUG
  if (deps.item_providers().count({_owner, name()}) == 0)
  {
    std::stringstream ss;
    ss << "Variable '" << name() << "' declared by model '" << owner().name()
       << "' has no provider in the dependency resolver. The dependency graph knows the providers "
          "for the following variables:\n";
    for (const auto & item : deps.item_providers())
      ss << "    " << item.first.parent->name() << ": " << item.first.value << '\n';
    ss << "and the dependency graph knows about the following leaf variables:\n";
    for (const auto & item : deps.inbound_items())
      ss << "    " << item.parent->name() << ": " << item.value << '\n';
    throw NEMLException(ss.str());
  }
#endif

  const auto providers = deps.item_providers().at({_owner, name()});
  neml_assert_dbg(
      providers.size() == 1, "Internal error: variable '", name(), "' has multiple providers.");
  return providers.begin()->parent->output_variable(name());
}

const VariableBase::DerivContainer &
VariableBase::total_derivatives(const DependencyResolver<Model, VariableName> & deps) const
{
  // return cached derivatives if available
  if (!_total_derivs.empty())
    return _total_derivs;

  for (const auto & [dy_du, uvar] : derivatives())
  {
    if (!dy_du.defined())
      continue;
    // u is leaf variable
    if (uvar->is_leaf(deps))
      get_deriv(_total_derivs,
                *this,
                *uvar,
                dy_du.intrsc_intmd_dim(),
                dy_du.var_intrsc_intmd_dim(),
                dy_du.arg_intrsc_intmd_dim(0)) += dy_du;
    // apply chain rule
    else
      for (const auto & [du_dx, xvar] : uvar->provider(deps).total_derivatives(deps))
      {
        auto dy_dx = chain_rule(dy_du, du_dx);
        get_deriv(_total_derivs,
                  *this,
                  *xvar,
                  dy_dx.intrsc_intmd_dim(),
                  dy_dx.var_intrsc_intmd_dim(),
                  dy_dx.arg_intrsc_intmd_dim(0)) += dy_dx;
      }
  }

  return _total_derivs;
}

const VariableBase::SecDerivContainer &
VariableBase::total_second_derivatives(const DependencyResolver<Model, VariableName> & deps) const
{
  // return cached derivatives if available
  if (!_total_sec_derivs.empty())
    return _total_sec_derivs;

  for (const auto & [d2y_du1u2, u1var, u2var] : second_derivatives())
  {
    if (!d2y_du1u2.defined())
      continue;

    // both u1 and u2 are leaf variables
    if (u1var->is_leaf(deps) && u2var->is_leaf(deps))
      get_secderiv(_total_sec_derivs,
                   *this,
                   *u1var,
                   *u2var,
                   d2y_du1u2.intrsc_intmd_dim(),
                   d2y_du1u2.var_intrsc_intmd_dim(),
                   d2y_du1u2.arg_intrsc_intmd_dim(0),
                   d2y_du1u2.arg_intrsc_intmd_dim(1)) += d2y_du1u2;

    // u1 is leaf, u2 is not
    else if (u1var->is_leaf(deps))
      for (const auto & [du2_dx2, x2var] : u2var->provider(deps).total_derivatives(deps))
      {
        auto d2y_dx1x2 = chain_rule(d2y_du1u2, nullptr, &du2_dx2);
        get_secderiv(_total_sec_derivs,
                     *this,
                     *u1var,
                     *x2var,
                     d2y_dx1x2.intrsc_intmd_dim(),
                     d2y_dx1x2.var_intrsc_intmd_dim(),
                     d2y_dx1x2.arg_intrsc_intmd_dim(0),
                     d2y_dx1x2.arg_intrsc_intmd_dim(1)) += d2y_dx1x2;
      }

    // u2 is leaf, u1 is not
    else if (u2var->is_leaf(deps))
      for (const auto & [du1_dx1, x1var] : u1var->provider(deps).total_derivatives(deps))
      {
        auto d2y_dx1x2 = chain_rule(d2y_du1u2, &du1_dx1, nullptr);
        get_secderiv(_total_sec_derivs,
                     *this,
                     *x1var,
                     *u2var,
                     d2y_dx1x2.intrsc_intmd_dim(),
                     d2y_dx1x2.var_intrsc_intmd_dim(),
                     d2y_dx1x2.arg_intrsc_intmd_dim(0),
                     d2y_dx1x2.arg_intrsc_intmd_dim(1)) += d2y_dx1x2;
      }

    // neither u1 nor u2 is leaf variable
    else
      for (const auto & [du1_dx1, x1var] : u1var->provider(deps).total_derivatives(deps))
        for (const auto & [du2_dx2, x2var] : u2var->provider(deps).total_derivatives(deps))
        {
          auto d2y_dx1x2 = chain_rule(d2y_du1u2, &du1_dx1, &du2_dx2);
          get_secderiv(_total_sec_derivs,
                       *this,
                       *x1var,
                       *x2var,
                       d2y_dx1x2.intrsc_intmd_dim(),
                       d2y_dx1x2.var_intrsc_intmd_dim(),
                       d2y_dx1x2.arg_intrsc_intmd_dim(0),
                       d2y_dx1x2.arg_intrsc_intmd_dim(1)) += d2y_dx1x2;
        }
  }

  for (const auto & [dy_du, uvar] : derivatives())
  {
    if (!dy_du.defined())
      continue;

    // second derivative of a leaf variable is zero
    if (uvar->is_leaf(deps))
      continue;

    for (const auto & [d2u_dx1x2, x1var, x2var] :
         uvar->provider(deps).total_second_derivatives(deps))
    {
      auto d2y_dx1x2 = chain_rule(dy_du, d2u_dx1x2);
      get_secderiv(_total_sec_derivs,
                   *this,
                   *x1var,
                   *x2var,
                   d2y_dx1x2.intrsc_intmd_dim(),
                   d2y_dx1x2.var_intrsc_intmd_dim(),
                   d2y_dx1x2.arg_intrsc_intmd_dim(0),
                   d2y_dx1x2.arg_intrsc_intmd_dim(1)) += d2y_dx1x2;
    }
  }

  return _total_sec_derivs;
}

void
VariableBase::clear_chain_rule_cache(const DependencyResolver<Model, VariableName> & deps) const
{
  _total_derivs.clear();
  _total_sec_derivs.clear();
  for (const auto & [dy_du, uvar] : derivatives())
    if (!uvar->is_leaf(deps))
      uvar->provider(deps).clear_chain_rule_cache(deps);
}
} // namespace neml2
