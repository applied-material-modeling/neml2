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

// Internal header -- NOT shipped. The residual/step provider the C++ Newton
// solver drives. Concrete backends decouple the solver from where residuals
// come from: `AOTINonlinearSystem` (compiled .pt2 loaders) and -- once the
// eager path is unified -- a Python-delegating system. Givens and any promoted
// parameters are bound at construction; only the unknowns vary across Newton
// iterations.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>

namespace neml2::aoti
{
/// Light per-group layout descriptor consumed by the Newton convergence /
/// line-search reductions. ``structure`` is "block" (sub_batch axes preserved)
/// or "dense" (sub_batch folded into the base axis). This is the subset of the
/// segment's per-group metadata the solver actually needs -- it is intentionally
/// independent of the AOTI ``Segment`` so the same solver can be driven by a
/// non-AOTI (eager) system.
struct GroupLayout
{
  std::string structure;
  std::vector<int64_t> sub_batch_shape;
};

/// Abstract residual/step provider. All tensors are per-group, following the
/// AssembledVector convention: BLOCK groups are ``(*B, *sub_batch, base_total)``
/// and DENSE groups are ``(*B, group_total)``.
class NonlinearSystem
{
public:
  virtual ~NonlinearSystem();

  /// Residual ``b = -r`` at the current unknowns, one tensor per residual
  /// group. Cheap relative to ``step``; called once per line-search trial.
  virtual std::vector<at::Tensor> residual(const std::vector<at::Tensor> & u) const = 0;

  /// One Newton step: ``(du_groups, b_groups)``. The assemble + linear solve
  /// are folded in (compiled into the graph on the AOTI path; a live
  /// assemble+solve on the eager path), so the solver never names a linear
  /// solver. ``du_groups`` has one tensor per unknown group; ``b_groups`` is
  /// the residual at the current point (returned for symmetry, may be unused).
  virtual std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
  step(const std::vector<at::Tensor> & u) const = 0;

  /// Per-group layouts (in declared order) for the unknown and residual sides.
  virtual const std::vector<GroupLayout> & unknown_layout() const = 0;
  virtual const std::vector<GroupLayout> & residual_layout() const = 0;
};
} // namespace neml2::aoti
