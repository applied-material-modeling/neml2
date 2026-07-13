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

#include "neml2/csrc/aoti/nonlinear_system_krylov_aoti.h"
#include "neml2/csrc/aoti/assertions.h"

#include <utility>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace neml2::aoti
{
KrylovAOTINonlinearSystem::KrylovAOTINonlinearSystem(
    torch::inductor::AOTIModelPackageLoader & residual_loader,
    torch::inductor::AOTIModelPackageLoader & matvec_loader,
    torch::inductor::AOTIModelPackageLoader * precond_setup_loader,
    torch::inductor::AOTIModelPackageLoader * precond_apply_loader,
    std::vector<GroupLayout> unknown_layout,
    std::vector<GroupLayout> residual_layout,
    std::vector<at::Tensor> g_groups,
    std::vector<at::Tensor> params,
    KrylovConfig cfg)
  : KrylovNonlinearSystem(std::move(unknown_layout), std::move(residual_layout), cfg),
    _residual_loader(residual_loader),
    _matvec_loader(matvec_loader),
    _precond_setup_loader(precond_setup_loader),
    _precond_apply_loader(precond_apply_loader),
    _g(std::move(g_groups)),
    _params(std::move(params))
{
}

std::vector<at::Tensor>
KrylovAOTINonlinearSystem::build_inputs(const std::vector<at::Tensor> & u) const
{
  std::vector<at::Tensor> v;
  v.reserve(u.size() + _g.size() + _params.size());
  for (const auto & t : u)
    v.push_back(t.contiguous());
  for (const auto & t : _g)
    v.push_back(t);
  for (const auto & t : _params)
    v.push_back(t);
  return v;
}

std::vector<at::Tensor>
KrylovAOTINonlinearSystem::residual_raw(const std::vector<at::Tensor> & u) const
{
  return _residual_loader.run(build_inputs(u));
}

std::vector<at::Tensor>
KrylovAOTINonlinearSystem::matvec_raw(const std::vector<at::Tensor> & u,
                                      const std::vector<at::Tensor> & v) const
{
  // The matvec graph signature is (*u, *g, *params, *v).
  auto inputs = build_inputs(u);
  inputs.reserve(inputs.size() + v.size());
  for (const auto & t : v)
    inputs.push_back(t.contiguous());
  return _matvec_loader.run(inputs);
}

std::vector<at::Tensor>
KrylovAOTINonlinearSystem::precond_setup_raw(const std::vector<at::Tensor> & u) const
{
  // The authored setup graph maps (*u, *g, *params) -> the preconditioner state
  // tensors (e.g. an LU + pivots, per-variable inverses, a diagonal reciprocal).
  _assert(_precond_setup_loader != nullptr,
          "KrylovAOTINonlinearSystem: no preconditioner setup graph");
  return _precond_setup_loader->run(build_inputs(u));
}

at::Tensor
KrylovAOTINonlinearSystem::precond_apply_raw(const std::vector<at::Tensor> & state,
                                             const at::Tensor & r_flat) const
{
  // The authored apply graph maps (*state, r_flat) -> z_flat = M^-1 r.
  _assert(_precond_apply_loader != nullptr,
          "KrylovAOTINonlinearSystem: no preconditioner apply graph");
  std::vector<at::Tensor> inputs;
  inputs.reserve(state.size() + 1);
  for (const auto & s : state)
    inputs.push_back(s);
  inputs.push_back(r_flat.contiguous());
  const auto outs = _precond_apply_loader->run(inputs);
  _assert(outs.size() == 1, "KrylovAOTINonlinearSystem: precond apply must return one tensor");
  return outs.front();
}
} // namespace neml2::aoti
