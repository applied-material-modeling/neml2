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

// Internal header -- NOT shipped. The `NonlinearSystem` whose `step()` runs a
// matrix-free Krylov linear solve (krylov.h) instead of a baked direct solve.
// The outer C++ `Newton::solve` drives it exactly like the direct system --
// `step()` returns an (inexact) `(du, b)` it treats opaquely.
//
// This base holds the shared `step()` (flatten -> Krylov -> unflatten, plus the
// preconditioner cache policy); the two concrete backends supply only the raw
// per-group operations:
//   - AOTI  (`KrylovAOTINonlinearSystem`): residual/matvec/jacobian .pt2 loaders
//   - eager (`KrylovEagerNonlinearSystem`): Python callbacks
// so the Krylov numerics + cache logic live in exactly one place (parity by
// construction, mirroring how the direct `Newton` loop is shared).
//
// Header-only (like krylov.h): each backend lives in a different shared object
// (libneml2.so vs the pyaoti module), and each instantiates the base within its
// own TU, so there is no cross-boundary polymorphism.

#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>

#include "neml2/csrc/aoti/assertions.h"
#include "neml2/csrc/aoti/krylov.h"
#include "neml2/csrc/aoti/krylov_flatten.h"
#include "neml2/csrc/aoti/nonlinear_system.h"

namespace neml2::aoti
{
class KrylovNonlinearSystem : public NonlinearSystem
{
public:
  /// `block_sizes` are the per-variable widths of the (single, dense) unknown
  /// group, used by the BlockJacobi preconditioner (ignored otherwise).
  KrylovNonlinearSystem(std::vector<GroupLayout> unknown_layout,
                        std::vector<GroupLayout> residual_layout,
                        KrylovConfig cfg,
                        std::vector<int64_t> block_sizes)
    : _unknown_layout(std::move(unknown_layout)),
      _residual_layout(std::move(residual_layout)),
      _cfg(cfg),
      _precond(cfg.precond, std::move(block_sizes))
  {
  }

  std::vector<at::Tensor> residual(const std::vector<at::Tensor> & u) const override
  {
    return residual_raw(u);
  }

  std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
  step(const std::vector<at::Tensor> & u) const override
  {
    assert_all_dense(_unknown_layout, "unknown");
    assert_all_dense(_residual_layout, "residual");

    // RHS b (= -r) for `A du = b`; flattened with the residual layout.
    auto b_groups = residual_raw(u);
    FlatSpec bspec;
    auto b_flat = flatten_dense(b_groups, _residual_layout, bspec);

    // Capture the unknown-side spec so the matvec/du round-trip is exact. The
    // Krylov vectors (v, du) live in the *batched* residual space, but the
    // initial guess u0 may be an unbatched broadcast (e.g. a predictor scalar,
    // shape (N,) not (*B, N)) -- the direct path only becomes batched after the
    // first u + du. Take the widths from the unknown groups but the batch from
    // the (authoritative, always-batched) residual so unflatten reshapes v/du to
    // (*B, w) rather than the batchless (w,).
    FlatSpec uspec;
    (void)flatten_dense(u, _unknown_layout, uspec);
    uspec.batch_shape = bspec.batch_shape;

    // Matrix-free operator A.v = dr/du . v at the current u (compiled/callback
    // matvec), flattened/unflattened at the boundary.
    const MatvecFn matvec = [&](const at::Tensor & v_flat) -> at::Tensor
    {
      auto v_groups = unflatten_dense(v_flat, uspec);
      auto jv_groups = matvec_raw(u, v_groups);
      FlatSpec tmp;
      return flatten_dense(jv_groups, _residual_layout, tmp);
    };

    // Preconditioner: (re)build per the cache policy, then apply.
    if (_cfg.precond != PrecondKind::None)
    {
      _assert(_unknown_layout.size() == 1,
              "preconditioned Krylov currently requires a single dense unknown "
              "group (v1); use preconditioner=none for multi-group systems.");
      if (needs_precond_rebuild())
        _precond.setup(assemble_dense_A(u));
    }
    const PrecondFn minv = [&](const at::Tensor & r) -> at::Tensor { return _precond.apply(r); };

    auto res = krylov_solve(matvec, minv, b_flat, _cfg);
    _last_iters = res.max_iters;

    return {unflatten_dense(res.du, uspec), std::move(b_groups)};
  }

  const std::vector<GroupLayout> & unknown_layout() const override { return _unknown_layout; }
  const std::vector<GroupLayout> & residual_layout() const override { return _residual_layout; }

protected:
  /// b = -r at u, one tensor per residual group.
  virtual std::vector<at::Tensor> residual_raw(const std::vector<at::Tensor> & u) const = 0;
  /// J.v = dr/du . v at u; `v` is per unknown group, result per residual group.
  virtual std::vector<at::Tensor> matvec_raw(const std::vector<at::Tensor> & u,
                                             const std::vector<at::Tensor> & v) const = 0;
  /// The assembled dense Jacobian `(Bflat, N, N)` at u -- only called when a
  /// preconditioner is configured (single dense group).
  virtual at::Tensor assemble_dense_A(const std::vector<at::Tensor> & u) const = 0;

private:
  // Cache policy: None rebuilds every step; Chord builds once and reuses;
  // QualityThreshold rebuilds when the last Krylov iteration count exceeded a bar.
  bool needs_precond_rebuild() const
  {
    switch (_cfg.cache)
    {
      case CacheStrategy::None:
        return true;
      case CacheStrategy::Chord:
        return !_precond.ready();
      case CacheStrategy::QualityThreshold:
        return !_precond.ready() || _last_iters > _cfg.quality_threshold;
    }
    return true;
  }

  std::vector<GroupLayout> _unknown_layout;
  std::vector<GroupLayout> _residual_layout;
  KrylovConfig _cfg;
  // Cache state lives one Newton solve: the system is constructed fresh per
  // `_run_implicit_segment` call, so `mutable` here resets naturally each solve.
  mutable Preconditioner _precond;
  mutable int64_t _last_iters = 0;
};
} // namespace neml2::aoti
