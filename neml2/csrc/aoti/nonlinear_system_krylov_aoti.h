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

// Internal header -- NOT shipped. The AOTI-backed matrix-free Krylov system: the
// concrete `KrylovNonlinearSystem` whose raw residual / matvec / dense-Jacobian
// ops run compiled `.pt2` loaders. The shared base owns the Krylov numerics +
// preconditioner cache; this only supplies the compiled graph calls. Givens and
// the promoted-parameter tail are bound once at construction; only the unknowns
// vary per iteration.

#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>

#include "neml2/csrc/aoti/krylov.h"
#include "neml2/csrc/aoti/nonlinear_system_krylov.h"

namespace torch::inductor
{
class AOTIModelPackageLoader;
}

namespace neml2::aoti
{
class KrylovAOTINonlinearSystem : public KrylovNonlinearSystem
{
public:
  /// `residual_loader` / `matvec_loader` must outlive this system (owned by the
  /// segment). `jacobian_loader` may be null (no preconditioner). `g_groups` are
  /// the per-group givens; `params` the promoted-parameter tail in graph-call
  /// order; `block_sizes` the per-variable widths of the (single dense) unknown
  /// group for BlockJacobi.
  KrylovAOTINonlinearSystem(torch::inductor::AOTIModelPackageLoader & residual_loader,
                            torch::inductor::AOTIModelPackageLoader & matvec_loader,
                            torch::inductor::AOTIModelPackageLoader * jacobian_loader,
                            std::vector<GroupLayout> unknown_layout,
                            std::vector<GroupLayout> residual_layout,
                            std::vector<at::Tensor> g_groups,
                            std::vector<at::Tensor> params,
                            KrylovConfig cfg,
                            std::vector<int64_t> block_sizes);

protected:
  std::vector<at::Tensor> residual_raw(const std::vector<at::Tensor> & u) const override;
  std::vector<at::Tensor> matvec_raw(const std::vector<at::Tensor> & u,
                                     const std::vector<at::Tensor> & v) const override;
  at::Tensor assemble_dense_A(const std::vector<at::Tensor> & u) const override;

private:
  /// Loader-call input list: `(*u_groups, *g_groups, *params)`.
  std::vector<at::Tensor> build_inputs(const std::vector<at::Tensor> & u) const;

  torch::inductor::AOTIModelPackageLoader & _residual_loader;
  torch::inductor::AOTIModelPackageLoader & _matvec_loader;
  torch::inductor::AOTIModelPackageLoader * _jacobian_loader;
  std::vector<at::Tensor> _g;
  std::vector<at::Tensor> _params;
};
} // namespace neml2::aoti
