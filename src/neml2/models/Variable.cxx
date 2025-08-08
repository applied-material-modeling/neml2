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
#include "neml2/tensors/tensors.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/bmm.h"
#include "neml2/jit/utils.h"
#include "neml2/jit/TraceableTensorShape.h"
#include <ATen/core/interned_strings.h>
#include <ATen/ops/permute.h>
#include <numeric>

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

Tensor
VariableBase::from_assembly(const Tensor & val) const
{
  // shortcut when there's no left-batch dimension
  if (!lbatch_dim())
    return val.base_reshape(base_sizes());

  // left-batch shapes are special:
  // it is a "base" shape at assembly time, and a "batch" shape at operation time
  const auto B = utils::add_traceable_shapes(lbatch_sizes(), val.batch_sizes());
  const auto D = utils::add_shapes(lbatch_sizes(), base_sizes());
  // reshape to (batch; lbatch, base)
  auto v = val.base_reshape(D);
  // move lbatch to the front, i.e. (lbatch, batch; base)
  TensorShape permutation(v.dim());
  std::iota(permutation.begin(), permutation.end(), 0);
  auto begin = permutation.begin() + val.batch_dim();
  auto end = begin + lbatch_dim();
  std::rotate(begin, end, permutation.begin());
  return Tensor(at::permute(v, permutation), B);
}

Tensor
VariableBase::to_assembly(const Tensor & val) const
{
  // shortcut when there's no left-batch dimension
  if (!lbatch_dim())
    return val.base_flatten();

  // variable format has shape (lbatch, batch; base)
  // we need to permute it to (batch; lbatch, base)
  TensorShape permutation(val.dim());
  std::iota(permutation.begin(), permutation.end(), 0);
  auto begin = permutation.begin();
  auto end = begin + lbatch_dim();
  std::rotate(begin, end, permutation.end() - base_dim());
  return Tensor(at::permute(val, permutation), val.batch_sizes().slice(lbatch_dim()))
      .base_flatten();
}

Derivative
VariableBase::d(const VariableBase & var)
{
  neml_assert_dbg(owning(),
                  "Cannot assign derivative to a referencing variable '",
                  name(),
                  "' with respect to '",
                  var.name(),
                  "'.");
  return Derivative(*this, var, &_derivs[var.name()]);
}

Derivative
VariableBase::d(const VariableBase & var1, const VariableBase & var2)
{
  neml_assert_dbg(owning(),
                  "Cannot assign second derivative to a referencing variable '",
                  name(),
                  "' with respect to '",
                  var1.name(),
                  "' and '",
                  var2.name(),
                  "'.");
  return Derivative(*this, var1, var2, &_sec_derivs[var1.name()][var2.name()]);
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
    neml_assert(u, "Cannot request AD for a null variable.");
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
  for (const auto & [model, var] : dep.outbound_items())
    if (var == name())
    {
      _derivs = total_derivatives(dep, model, var);
      return;
    }
}

void
VariableBase::apply_second_order_chain_rule(const DependencyResolver<Model, VariableName> & dep)
{
  for (const auto & [model, var] : dep.outbound_items())
    if (var == name())
    {
      _sec_derivs = total_second_derivatives(dep, model, var);
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
                                const VariableName & yvar) const
{
  ValueMap derivs;

  for (const auto & [uvar, dy_du] : model->output_variable(yvar).derivatives())
  {
    if (dep.inbound_items().count({model, uvar}))
      assign_or_add(derivs[uvar], dy_du);
    else
      for (const auto & depu : dep.item_providers().at({model, uvar}))
        for (const auto & [xvar, du_dx] : total_derivatives(dep, depu.parent, uvar))
          assign_or_add(derivs[xvar], bmm(dy_du, du_dx));
  }

  return derivs;
}

DerivMap
VariableBase::total_second_derivatives(const DependencyResolver<Model, VariableName> & dep,
                                       Model * model,
                                       const VariableName & yvar) const
{
  DerivMap sec_derivs;

  for (const auto & [u1var, d2y_du1] : model->output_variable(yvar).second_derivatives())
    for (const auto & [u2var, d2y_du1u2] : d2y_du1)
    {
      if (dep.inbound_items().count({model, u1var}) && dep.inbound_items().count({model, u2var}))
        assign_or_add(sec_derivs[u1var][u2var], d2y_du1u2);
      else if (dep.inbound_items().count({model, u1var}))
        for (const auto & depu2 : dep.item_providers().at({model, u2var}))
          for (const auto & [x2var, du2_dxk] : total_derivatives(dep, depu2.parent, u2var))
            assign_or_add(sec_derivs[u1var][x2var],
                          Tensor(at::einsum("...ijq,...qk", {d2y_du1u2, du2_dxk}),
                                 utils::broadcast_batch_dim(d2y_du1u2, du2_dxk)));
      else if (dep.inbound_items().count({model, u2var}))
        for (const auto & depu1 : dep.item_providers().at({model, u1var}))
          for (const auto & [x1var, du1_dxj] : total_derivatives(dep, depu1.parent, u1var))
            assign_or_add(sec_derivs[x1var][u2var],
                          Tensor(at::einsum("...ipk,...pj", {d2y_du1u2, du1_dxj}),
                                 utils::broadcast_batch_dim(d2y_du1u2, du1_dxj)));
      else
        for (const auto & depu1 : dep.item_providers().at({model, u1var}))
          for (const auto & [x1var, du1_dxj] : total_derivatives(dep, depu1.parent, u1var))
            for (const auto & depu2 : dep.item_providers().at({model, u2var}))
              for (const auto & [x2var, du2_dxk] : total_derivatives(dep, depu2.parent, u2var))
                assign_or_add(
                    sec_derivs[x1var][x2var],
                    Tensor(at::einsum("...ipq,...pj,...qk", {d2y_du1u2, du1_dxj, du2_dxk}),
                           utils::broadcast_batch_dim(d2y_du1u2, du1_dxj, du2_dxk)));
    }

  for (const auto & [uvar, dy_du] : model->output_variable(yvar).derivatives())
    if (!dep.inbound_items().count({model, uvar}))
      for (const auto & depu : dep.item_providers().at({model, uvar}))
        for (const auto & [x1var, d2u_dx1] : total_second_derivatives(dep, depu.parent, uvar))
          for (const auto & [x2var, d2u_dx1x2] : d2u_dx1)
            assign_or_add(sec_derivs[x1var][x2var],
                          Tensor(at::einsum("...ip,...pjk", {dy_du, d2u_dx1x2}),
                                 utils::broadcast_batch_dim(dy_du, d2u_dx1x2)));

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
    _value = from_assembly(val);
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
void
Variable<T>::set(const ATensor & val, bool force)
{
  if (owning())
  {
    if constexpr (std::is_same_v<T, Tensor>)
      _value = T(val, val.dim() - base_dim());
    else
      _value = T(val);
  }
  else
  {
    neml_assert_dbg(_ref_is_mutable || force,
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
  return to_assembly(_value);
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
    _value = T(val);
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

Derivative::Derivative(const VariableBase & var1, const VariableBase & var2, Tensor * deriv)
  : _lbatch_sizes({var1.lbatch_sizes(), var2.lbatch_sizes()}),
    _base_sizes({var1.base_sizes(), var2.base_sizes()}),
    _assembly_sizes({var1.assembly_storage(), var2.assembly_storage()}),
    _deriv(deriv)
#ifndef NDEBUG
    ,
    _debug_name(std::string("d(") + var1.name().str() + ")/d(" + var2.name().str() + ")")
#endif
{
}

Derivative::Derivative(const VariableBase & var1,
                       const VariableBase & var2,
                       const VariableBase & var3,
                       Tensor * deriv)
  : _lbatch_sizes({var1.lbatch_sizes(), var2.lbatch_sizes(), var3.lbatch_sizes()}),
    _base_sizes({var1.base_sizes(), var2.base_sizes(), var3.base_sizes()}),
    _assembly_sizes({var1.assembly_storage(), var2.assembly_storage(), var3.assembly_storage()}),
    _deriv(deriv)
#ifndef NDEBUG
    ,
    _debug_name(std::string("d2(") + var1.name().str() + ")/d(" + var2.name().str() + ")/d(" +
                var3.name().str() + ")")
#endif
{
}

Derivative &
Derivative::operator=(const Tensor & val)
{
#ifndef NDEBUG
  // check if the given derivative has the correct left-batch shape
  const auto lbatch_sizes =
      _lbatch_sizes.size() == 2
          ? utils::add_shapes(_lbatch_sizes[0], _lbatch_sizes[1])
          : utils::add_shapes(_lbatch_sizes[0], _lbatch_sizes[1], _lbatch_sizes[2]);
  neml_assert_dbg(val.batch_dim() >= Size(lbatch_sizes.size()) &&
                      val.batch_sizes().slice(0, Size(lbatch_sizes.size())) == lbatch_sizes,
                  "The assigned derivative for ",
                  _debug_name,
                  " has incorrect batch shape ",
                  val.batch_sizes(),
                  " which is incompatible with the expected left-batch shape of ",
                  lbatch_sizes,
                  ".");
  // check if the given derivative has the correct base shape
  const auto base_sizes = _base_sizes.size() == 2
                              ? utils::add_shapes(_base_sizes[0], _base_sizes[1])
                              : utils::add_shapes(_base_sizes[0], _base_sizes[1], _base_sizes[2]);
  const auto base_dim = Size(base_sizes.size());
  neml_assert_dbg(val.base_dim() == base_dim && val.base_sizes() == base_sizes,
                  "The assigned derivative for ",
                  _debug_name,
                  " has incorrect base shape ",
                  val.base_sizes(),
                  " which is incompatible with the expected base shape of ",
                  base_sizes,
                  ".");
#endif

  // shortcut when there's no left-batch dimension
  if (_lbatch_sizes.empty())
  {
    assign_or_add(*_deriv, val.base_reshape(_assembly_sizes));
    return *this;
  }

  // Move each left-batch to the base
  //
  // For example, for first order derivatives with left-batch shapes, the tensor shape of the given
  // derivative is
  //   (lbatch1, lbatch2, batch; base1, base2)
  //
  // We first move lbatch1 before base1:
  //   (lbatch2, batch; lbatch1, base1, base2)
  //
  // Then we move lbatch2 before base2:
  //   (batch; lbatch1, base1, lbatch2, base2)
  //
  // Finally, we can flatten the base dimensions to get the assembly shape:
  //   (batch; assembly1, assembly2)
  //
  // The same procedure applies to second order derivatives with three left-batch groups.
  Size lbatch_dim = 0;
  TensorShape indices(val.dim());
  std::iota(indices.begin(), indices.end(), 0);
  auto permutation = indices;
  auto itr = permutation.begin() + val.batch_dim();
  for (std::size_t i = 0; i < _lbatch_sizes.size(); ++i)
  {
    if (!_lbatch_sizes[i].empty())
    {
      lbatch_dim += Size(_lbatch_sizes[i].size());
      auto begin = permutation.begin();
      auto end = begin + _lbatch_sizes[i].size();
      std::rotate(begin, end, itr);
    }
    itr = permutation.begin() + _base_sizes[i].size();
  }

  auto B = val.batch_sizes().slice(lbatch_dim);
  auto val2 = Tensor(at::permute(val, permutation), B);
  assign_or_add(*_deriv, val2.base_reshape(_assembly_sizes));
  return *this;
}

Derivative &
Derivative::operator=(const VariableBase & var)
{
  return Derivative::operator=(var.tensor());
}
}
