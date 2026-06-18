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

#pragma once

// Internal header -- NOT shipped. The AOTI-backed `NonlinearSystem`: residual /
// step come from a compiled implicit segment's `.pt2` loaders. Deliberately
// decoupled from the private `Model::Impl`/`Segment` -- the owning Impl member
// builds the layouts and hands over the two loaders, so this class depends only
// on the loader type and `NonlinearSystem`. Givens and the promoted-parameter
// tail are bound once at construction (constant across the solve); only the
// unknowns vary per iteration.

#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>

#include "neml2/csrc/aoti/nonlinear_system.h"

namespace torch::inductor
{
class AOTIModelPackageLoader;
}

namespace neml2::aoti
{
class AOTINonlinearSystem : public NonlinearSystem
{
public:
  /// ``rhs_loader``/``step_loader`` must outlive this system (owned by the
  /// segment). ``g_groups`` are the per-group given tensors; ``params`` is the
  /// promoted-parameter tail in graph-call order.
  AOTINonlinearSystem(torch::inductor::AOTIModelPackageLoader & rhs_loader,
                      torch::inductor::AOTIModelPackageLoader & step_loader,
                      std::vector<GroupLayout> unknown_layout,
                      std::vector<GroupLayout> residual_layout,
                      std::vector<at::Tensor> g_groups,
                      std::vector<at::Tensor> params);

  std::vector<at::Tensor> residual(const std::vector<at::Tensor> & u) const override;
  std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
  step(const std::vector<at::Tensor> & u) const override;
  const std::vector<GroupLayout> & unknown_layout() const override { return _unknown_layout; }
  const std::vector<GroupLayout> & residual_layout() const override { return _residual_layout; }

private:
  /// Loader-call input list: ``(*u_groups, *g_groups, *params)``.
  std::vector<at::Tensor> build_inputs(const std::vector<at::Tensor> & u) const;

  torch::inductor::AOTIModelPackageLoader & _rhs_loader;
  torch::inductor::AOTIModelPackageLoader & _step_loader;
  std::vector<GroupLayout> _unknown_layout;
  std::vector<GroupLayout> _residual_layout;
  std::vector<at::Tensor> _g;
  std::vector<at::Tensor> _params;
};
} // namespace neml2::aoti
