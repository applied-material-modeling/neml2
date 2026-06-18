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

#include "neml2/csrc/aoti/nonlinear_system_aoti.h"
#include "neml2/csrc/aoti/assertions.h"

#include <cstddef>
#include <utility>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace neml2::aoti
{
// Out-of-line dtor anchors the NonlinearSystem vtable in this TU.
NonlinearSystem::~NonlinearSystem() = default;

AOTINonlinearSystem::AOTINonlinearSystem(torch::inductor::AOTIModelPackageLoader & rhs_loader,
                                         torch::inductor::AOTIModelPackageLoader & step_loader,
                                         std::vector<GroupLayout> unknown_layout,
                                         std::vector<GroupLayout> residual_layout,
                                         std::vector<at::Tensor> g_groups,
                                         std::vector<at::Tensor> params)
  : _rhs_loader(rhs_loader),
    _step_loader(step_loader),
    _unknown_layout(std::move(unknown_layout)),
    _residual_layout(std::move(residual_layout)),
    _g(std::move(g_groups)),
    _params(std::move(params))
{
}

std::vector<at::Tensor>
AOTINonlinearSystem::build_inputs(const std::vector<at::Tensor> & u) const
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
AOTINonlinearSystem::residual(const std::vector<at::Tensor> & u) const
{
  return _rhs_loader.run(build_inputs(u));
}

std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
AOTINonlinearSystem::step(const std::vector<at::Tensor> & u) const
{
  // step graph returns (*du_groups, *b_curr_groups) of length n_u + n_r.
  const auto step_outs = _step_loader.run(build_inputs(u));
  const std::size_t n_u = _unknown_layout.size();
  const std::size_t n_r = _residual_layout.size();
  _assert(step_outs.size() == n_u + n_r,
          "AOTINonlinearSystem::step: step loader returned ",
          step_outs.size(),
          " tensors, expected n_unknown_groups + n_residual_groups (",
          n_u + n_r,
          ")");
  std::vector<at::Tensor> du(step_outs.begin(), step_outs.begin() + n_u);
  std::vector<at::Tensor> b(step_outs.begin() + n_u, step_outs.end());
  return {std::move(du), std::move(b)};
}
} // namespace neml2::aoti
