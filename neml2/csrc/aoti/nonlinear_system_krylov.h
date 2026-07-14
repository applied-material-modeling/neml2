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

#include <sstream>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>

#include "neml2/csrc/aoti/assertions.h"
#include "neml2/csrc/aoti/krylov.h"
#include "neml2/csrc/aoti/krylov_flatten.h"
#include "neml2/csrc/aoti/log.h"
#include "neml2/csrc/aoti/nonlinear_system.h"

namespace neml2::aoti
{
class KrylovNonlinearSystem : public NonlinearSystem
{
public:
  KrylovNonlinearSystem(std::vector<GroupLayout> unknown_layout,
                        std::vector<GroupLayout> residual_layout,
                        KrylovConfig cfg)
    : _unknown_layout(std::move(unknown_layout)),
      _residual_layout(std::move(residual_layout)),
      _cfg(cfg)
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

    // Preconditioner: a pair of authored setup/apply ops (compiled graphs on the
    // AOTI route, Python callbacks on the eager route) supplied by the subclass.
    // `precond_setup_raw` builds the state (factored/inverted from the model, e.g.
    // an LU or per-variable inverses) per the cache policy; `precond_apply_raw`
    // applies M^-1 to the flat residual. Absent (`has_preconditioner()==false`)
    // => identity (unpreconditioned).
    PrecondFn minv = [](const at::Tensor & r) -> at::Tensor { return r; };
    if (has_preconditioner())
    {
      if (needs_precond_rebuild())
      {
        _precond_state = precond_setup_raw(u);
        _precond_ready = true;
      }
      minv = [&](const at::Tensor & r) -> at::Tensor
      { return precond_apply_raw(_precond_state, r); };
    }

    // Inner linear-solve residual trace on the `linear` channel (debug). Built
    // only when the channel is on, so a silent solve incurs no per-iteration sync.
    // LCOV_EXCL_START -- diagnostic linear-residual trace (verbosity-gated; see neml2.log)
    LinearLogFn on_iter;
    if (log::enabled(log::Channel::Linear, log::Level::Debug))
      on_iter = [](int64_t it, const at::Tensor & resid, const at::Tensor & bnorm)
      {
        std::ostringstream oss;
        oss << "  krylov iter=" << it << " max|r|=" << std::scientific << resid.max().item<double>()
            << " max|r|/|b|=" << std::scientific << (resid / bnorm).max().item<double>();
        log::emit(log::Channel::Linear, log::Level::Debug, oss.str());
      };
    // LCOV_EXCL_STOP

    auto res = krylov_solve(matvec, minv, b_flat, _cfg, on_iter);
    _last_iters = res.max_iters;

    // Linear-solve convergence summary on the `linear` channel (mirrors the
    // Newton summary). A non-convergence -- some element hit max_its without
    // reaching tol, which degrades the Newton step -- surfaces at `warning` (shown
    // by default); the healthy summary rides `info`. Gated on `warning` so a
    // silent channel never pays the convergence-mask d2h sync.
    // LCOV_EXCL_START -- diagnostic linear-solve summary
    if (log::enabled(log::Channel::Linear, log::Level::Warning))
    {
      const bool all_converged = res.converged.defined() ? res.converged.all().item<bool>() : true;
      if (!all_converged)
      {
        const int64_t nfail = at::logical_not(res.converged).sum().item<int64_t>();
        std::ostringstream oss;
        oss << "linear solve NOT converged (iters=" << res.max_iters << ", " << nfail
            << " element(s) at max_its)";
        log::emit(log::Channel::Linear, log::Level::Warning, oss.str());
      }
      else if (log::enabled(log::Channel::Linear, log::Level::Info))
      {
        std::ostringstream oss;
        oss << "linear solve: iters=" << res.max_iters << ", converged";
        log::emit(log::Channel::Linear, log::Level::Info, oss.str());
      }
    }
    // LCOV_EXCL_STOP

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
  /// Whether a preconditioner is configured (its setup/apply ops are present).
  virtual bool has_preconditioner() const = 0;
  /// Build the preconditioner state from the model at u (authored setup graph /
  /// callback) -- the raw tensors `precond_apply_raw` consumes. Only called when
  /// `has_preconditioner()` and the cache policy requests a rebuild.
  virtual std::vector<at::Tensor> precond_setup_raw(const std::vector<at::Tensor> & u) const = 0;
  /// Apply M^-1 to the flat `(Bflat, N)` residual using the cached state.
  virtual at::Tensor precond_apply_raw(const std::vector<at::Tensor> & state,
                                       const at::Tensor & r_flat) const = 0;

private:
  // Cache policy: None rebuilds every step; Chord builds once and reuses;
  // MaxLinearIters rebuilds when the last solve's iteration count exceeded a bar.
  bool needs_precond_rebuild() const
  {
    switch (_cfg.cache)
    {
      case CacheStrategy::None:
        return true;
      case CacheStrategy::Chord:
        return !_precond_ready;
      case CacheStrategy::MaxLinearIters:
        return !_precond_ready || _last_iters > _cfg.cache_max_its;
    }
    return true;
  }

  std::vector<GroupLayout> _unknown_layout;
  std::vector<GroupLayout> _residual_layout;
  KrylovConfig _cfg;
  // Cache state lives one Newton solve: the system is constructed fresh per
  // `_run_implicit_segment` call, so `mutable` here resets naturally each solve.
  mutable std::vector<at::Tensor> _precond_state;
  mutable bool _precond_ready = false;
  mutable int64_t _last_iters = 0;
};
} // namespace neml2::aoti
