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

// Internal header -- NOT shipped. Batched matrix-free Krylov linear solvers
// (restarted GMRES(m) + BiCGStab) operating on a flat `(B, N)` right-hand side,
// where `B` is the flattened dynamic batch and `N` the per-element unknown count.
// The operator `A` and the preconditioner `M^-1` are supplied as functors
// `(B, N) -> (B, N)`, so the same solver drives the AOTI route (matvec via a
// compiled `_matvec` loader) and the eager route (matvec via a Python callback).
//
// Header-only (like batch_chunk.h) so the two `NonlinearSystem` subclasses AND
// the C++ unit test can share it without adding an exported symbol -- it keeps
// the libneml2.so ABI surface limited to the public Model / scheduler classes.
//
// This is a faithful port of the validated Phase-B spike prototype
// (`benchmark/spike_gmres.py::batched_gmres` / `batched_bicgstab`): CGS2
// orthogonalization, batched Givens-rotation QR, `linalg_solve_triangular`
// back-substitution. The only control flow is host-side scalar loops plus a
// single `done.all().item<bool>()` restart test per cycle (the same class of
// device->host sync the Newton loop already incurs).

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

namespace neml2::aoti
{
/// Which Krylov method to run.
enum class KrylovMethod
{
  GMRES,
  BiCGStab,
};

/// Which preconditioner `M^-1 ~= A^-1` to apply (built from the assembled `A`;
/// consumed by the `NonlinearSystem` layer, not by the bare solvers here).
enum class PrecondKind
{
  None,
  Jacobi,
  BlockJacobi,
  Full,
};

/// How often the preconditioner setup (assemble + factor `A`) is rebuilt across
/// the outer Newton iterations (consumed by the `NonlinearSystem` layer).
enum class CacheStrategy
{
  None,             // rebuild every Newton step
  Chord,            // build once at the first step, reuse
  QualityThreshold, // rebuild when the last Krylov iteration count exceeds a bar
};

/// Iterative linear-solver tunables. `restart` is the GMRES(m) width; `max_iters`
/// is the total inner-iteration (matvec) budget shared by both methods.
struct KrylovConfig
{
  KrylovMethod method = KrylovMethod::GMRES;
  PrecondKind precond = PrecondKind::None;
  CacheStrategy cache = CacheStrategy::None;
  int64_t restart = 40;
  int64_t max_iters = 1000;
  double abs_tol = 0.0; // absolute Krylov residual stop (0 = relative only)
  double rel_tol = 1.0e-8;
  int64_t quality_threshold = 0; // for CacheStrategy::QualityThreshold
};

/// String tags (as they appear in the input file / metadata) -> enums. Unknown
/// values are a configuration error (a FatalError at the call site, which owns
/// the assertion helpers -- these headers stay dependency-light).
inline bool
parse_krylov_method(const std::string & s, KrylovMethod & out)
{
  if (s == "gmres")
    out = KrylovMethod::GMRES;
  else if (s == "bicgstab")
    out = KrylovMethod::BiCGStab;
  else
    return false;
  return true;
}

inline bool
parse_precond_kind(const std::string & s, PrecondKind & out)
{
  if (s == "none")
    out = PrecondKind::None;
  else if (s == "jacobi")
    out = PrecondKind::Jacobi;
  else if (s == "block_jacobi")
    out = PrecondKind::BlockJacobi;
  else if (s == "full")
    out = PrecondKind::Full;
  else
    return false;
  return true;
}

inline bool
parse_cache_strategy(const std::string & s, CacheStrategy & out)
{
  if (s == "none")
    out = CacheStrategy::None;
  else if (s == "chord")
    out = CacheStrategy::Chord;
  else if (s == "quality_threshold")
    out = CacheStrategy::QualityThreshold;
  else
    return false;
  return true;
}

/// `A x = b` operator / preconditioner applied to a flat `(B, N)` batch of
/// column vectors, returning `(B, N)`.
using MatvecFn = std::function<at::Tensor(const at::Tensor &)>;
using PrecondFn = std::function<at::Tensor(const at::Tensor &)>;

/// Result of an inner Krylov solve.
struct KrylovResult
{
  at::Tensor du;         ///< the solution `x`, shape `(B, N)`
  int64_t max_iters = 0; ///< max-over-batch inner iterations taken (matvecs)
  at::Tensor converged;  ///< per-element bool mask, shape `(B,)`
};

namespace detail
{
/// Machine epsilon for the tensor's floating dtype (defaults to double).
inline double
dtype_eps(at::ScalarType st)
{
  return st == at::kFloat ? static_cast<double>(std::numeric_limits<float>::epsilon())
                          : std::numeric_limits<double>::epsilon();
}

/// L2 norm along the last axis: `(B, N) -> (B,)`.
inline at::Tensor
row_norm(const at::Tensor & x)
{
  return (x * x).sum(-1).sqrt();
}

/// Floor a *signed* denominator's magnitude at `eps` while preserving its sign
/// (a zero sign floors to +eps), so a divide can never blow up yet the sign is
/// never flipped. Used for BiCGStab's inner-product denominators, which -- unlike
/// GMRES's norms -- can be negative (a plain `clamp_min` would corrupt them).
inline at::Tensor
safe_denom(const at::Tensor & d, double eps)
{
  auto sign = at::sign(d);
  sign = at::where(sign == 0, at::ones_like(sign), sign);
  return at::where(d.abs() < eps, sign * eps, d);
}
} // namespace detail

/// Restarted GMRES(m) with classical Gram-Schmidt (CGS2) reorthogonalization and
/// batched Givens-rotation QR. Solves `A x = b` (left-preconditioned by `minv`)
/// for every element of the leading batch simultaneously to a per-element
/// relative/absolute residual tolerance.
inline KrylovResult
gmres(const MatvecFn & matvec,
      const PrecondFn & minv,
      const at::Tensor & b,
      const KrylovConfig & cfg)
{
  const auto B = b.size(0);
  const auto n = b.size(1);
  const auto opts = b.options();
  const double eps = detail::dtype_eps(b.scalar_type());
  const int64_t m = cfg.restart;
  const int64_t max_restarts = (cfg.max_iters + m - 1) / m;

  auto x = at::zeros({B, n}, opts);
  const auto bnorm = detail::row_norm(b).clamp_min(eps); // (B,)
  auto iters = at::zeros({B}, opts.dtype(at::kLong));
  auto done = at::zeros({B}, opts.dtype(at::kBool));

  for (int64_t restart = 0; restart < max_restarts; ++restart)
  {
    auto r = minv(b - matvec(x));
    auto beta = detail::row_norm(r); // (B,)
    done = at::logical_or(done, at::logical_or(beta < cfg.abs_tol, beta / bnorm < cfg.rel_tol));
    if (done.all().item<bool>())
      break;

    auto V = at::zeros({B, m + 1, n}, opts);
    auto H = at::zeros({B, m + 1, m}, opts);
    auto cs = at::zeros({B, m}, opts);
    auto sn = at::zeros({B, m}, opts);
    auto g = at::zeros({B, m + 1}, opts);
    g.select(1, 0).copy_(beta);
    V.select(1, 0).copy_(r / beta.clamp_min(eps).unsqueeze(-1));
    const auto active = at::logical_not(done);

    for (int64_t j = 0; j < m; ++j)
    {
      auto w = minv(matvec(V.select(1, j))); // (B, n)
      // CGS2: classical Gram-Schmidt + one reorthogonalization (BLAS-2 friendly).
      for (int reorth = 0; reorth < 2; ++reorth)
      {
        auto Vj = V.slice(1, 0, j + 1);              // (B, j+1, n)
        auto hj = at::einsum("bkn,bn->bk", {Vj, w}); // (B, j+1)
        w = w - at::einsum("bk,bkn->bn", {hj, Vj});  // (B, n)
        H.slice(1, 0, j + 1).select(2, j).add_(hj);
      }
      auto hjp = detail::row_norm(w); // (B,)
      H.select(1, j + 1).select(1, j).copy_(hjp);
      V.select(1, j + 1).copy_(w / hjp.clamp_min(eps).unsqueeze(-1));

      // Apply the previous Givens rotations to the new column of H.
      for (int64_t i = 0; i < j; ++i)
      {
        auto Hij = H.select(1, i).select(1, j);      // (B,) view of H[:, i, j]
        auto Hi1j = H.select(1, i + 1).select(1, j); // (B,) view of H[:, i+1, j]
        auto ci = cs.select(1, i);
        auto si = sn.select(1, i);
        auto t0 = ci * Hij + si * Hi1j;
        auto t1 = -si * Hij + ci * Hi1j;
        Hij.copy_(t0);
        Hi1j.copy_(t1);
      }
      // New Givens rotation zeroing H[j+1, j].
      auto Hjj = H.select(1, j).select(1, j);      // (B,) view of H[:, j, j]
      auto Hj1j = H.select(1, j + 1).select(1, j); // (B,) view of H[:, j+1, j]
      auto denom = (Hjj * Hjj + Hj1j * Hj1j).sqrt().clamp_min(eps);
      cs.select(1, j).copy_(Hjj / denom);
      sn.select(1, j).copy_(Hj1j / denom);
      auto new_hjj = cs.select(1, j) * Hjj + sn.select(1, j) * Hj1j;
      Hjj.copy_(new_hjj);
      Hj1j.zero_();

      // Update the transformed RHS g.
      auto gj = g.select(1, j);      // (B,) view of g[:, j] (old)
      auto gj1 = g.select(1, j + 1); // (B,) view of g[:, j+1]
      auto new_gj1 = -sn.select(1, j) * gj;
      auto new_gj = cs.select(1, j) * gj;
      gj1.copy_(new_gj1);
      gj.copy_(new_gj);

      auto resid = g.select(1, j + 1).abs(); // (B,) current absolute residual
      iters += at::logical_and(active, at::logical_not(done)).to(at::kLong);
      auto newly =
          at::logical_and(at::logical_and(active, at::logical_not(done)),
                          at::logical_or(resid < cfg.abs_tol, resid / bnorm < cfg.rel_tol));
      done = at::logical_or(done, newly);
    }

    // Back-substitute R y = g on the full [0:m] window per element; x += V y.
    auto R = H.slice(1, 0, m).slice(2, 0, m);  // (B, m, m)
    auto rhs = g.slice(1, 0, m).unsqueeze(-1); // (B, m, 1)
    auto R_reg = R + eps * at::eye(m, opts);
    auto y = at::linalg_solve_triangular(R_reg,
                                         rhs,
                                         /*upper=*/true,
                                         /*left=*/true,
                                         /*unitriangular=*/false)
                 .squeeze(-1); // (B, m)
    x = x + at::einsum("bjn,bj->bn", {V.slice(1, 0, m), y});
  }

  return {x, iters.max().item<int64_t>(), done};
}

/// Stabilized biconjugate gradient (BiCGStab), left-preconditioned by `minv`.
/// Short-recurrence (no growing Krylov basis). Signed inner-product denominators
/// get a sign-preserving floor (`safe_denom`); converged elements are frozen
/// each iteration so a batch member that has already reached ~0 residual cannot
/// be corrupted by the ongoing (0/0) updates of the still-iterating members.
inline KrylovResult
bicgstab(const MatvecFn & matvec,
         const PrecondFn & minv,
         const at::Tensor & b,
         const KrylovConfig & cfg)
{
  const auto B = b.size(0);
  const auto n = b.size(1);
  const auto opts = b.options();
  const double eps = detail::dtype_eps(b.scalar_type());

  auto x = at::zeros({B, n}, opts);
  auto r = b - matvec(x);
  auto rhat = r.clone();
  const auto bnorm = detail::row_norm(b).clamp_min(eps);
  auto rho = at::ones({B}, opts);
  auto alpha = at::ones({B}, opts);
  auto omega = at::ones({B}, opts);
  auto v = at::zeros({B, n}, opts);
  auto p = at::zeros({B, n}, opts);
  auto iters = at::zeros({B}, opts.dtype(at::kLong));
  const auto stop = [&](const at::Tensor & rn)
  { return at::logical_or(rn < cfg.abs_tol, rn / bnorm < cfg.rel_tol); };
  auto done = stop(detail::row_norm(r));

  const auto dot = [](const at::Tensor & a, const at::Tensor & c)
  { return at::einsum("bn,bn->b", {a, c}); };

  for (int64_t it = 0; it < cfg.max_iters; ++it)
  {
    if (done.all().item<bool>())
      break;
    auto rho_new = dot(rhat, r);
    // rho, omega, dot(rhat,v) are signed inner products -> sign-preserving floor;
    // dot(t,t) is a non-negative norm^2 -> a plain clamp_min is correct.
    auto beta = (rho_new / detail::safe_denom(rho, eps)) * (alpha / detail::safe_denom(omega, eps));
    auto p_new = r + beta.unsqueeze(-1) * (p - omega.unsqueeze(-1) * v);
    auto phat = minv(p_new);
    auto v_new = matvec(phat);
    auto alpha_new = rho_new / detail::safe_denom(dot(rhat, v_new), eps);
    auto s = r - alpha_new.unsqueeze(-1) * v_new;
    auto shat = minv(s);
    auto t = matvec(shat);
    auto omega_new = dot(t, s) / dot(t, t).clamp_min(eps);
    auto x_new = x + alpha_new.unsqueeze(-1) * phat + omega_new.unsqueeze(-1) * shat;
    auto r_new = s - omega_new.unsqueeze(-1) * t;

    // Freeze converged elements: keep their carried state (so a done member's
    // 0/0 update is discarded rather than poisoning its solution).
    const auto keepV = done.unsqueeze(-1);
    x = at::where(keepV, x, x_new);
    r = at::where(keepV, r, r_new);
    p = at::where(keepV, p, p_new);
    v = at::where(keepV, v, v_new);
    rho = at::where(done, rho, rho_new);
    alpha = at::where(done, alpha, alpha_new);
    omega = at::where(done, omega, omega_new);

    iters += at::logical_not(done).to(at::kLong);
    done = at::logical_or(done, stop(detail::row_norm(r)));
  }

  return {x, iters.max().item<int64_t>(), done};
}

/// Dispatch to the configured method.
inline KrylovResult
krylov_solve(const MatvecFn & matvec,
             const PrecondFn & minv,
             const at::Tensor & b,
             const KrylovConfig & cfg)
{
  return cfg.method == KrylovMethod::BiCGStab ? bicgstab(matvec, minv, b, cfg)
                                              : gmres(matvec, minv, b, cfg);
}

/// A left preconditioner `M^-1 ~= A^-1`, built from the assembled dense operator
/// `A` (shape `(B, N, N)`) and applied to a flat `(B, N)` residual. Holds its
/// factored/inverted state so the `NonlinearSystem` layer can cache it across
/// Newton iterations (chord / quality-threshold strategies). A port of the
/// spike's `make_preconditioners`, using the `_ex` linalg variants so a singular
/// batch member yields an `info != 0` flag rather than throwing for the whole
/// batch (the offending row's inexact step is then caught by the outer Newton's
/// divergence check).
class Preconditioner
{
public:
  Preconditioner() = default;
  Preconditioner(PrecondKind kind, std::vector<int64_t> block_sizes)
    : _kind(kind),
      _block_sizes(std::move(block_sizes))
  {
  }

  PrecondKind kind() const { return _kind; }
  /// None is always "ready" (identity, no setup); the others need `setup`.
  bool ready() const { return _kind == PrecondKind::None || _ready; }
  void invalidate() { _ready = false; }

  /// Build `M^-1` from the dense operator `A` (`(B, N, N)`).
  void setup(const at::Tensor & A)
  {
    const double eps = detail::dtype_eps(A.scalar_type());
    switch (_kind)
    {
      case PrecondKind::None:
        break;
      case PrecondKind::Jacobi:
        // Reciprocal of the diagonal, sign-preserving floor so a ~0 pivot cannot
        // produce an inf apply.
        _diag = detail::safe_denom(at::diagonal(A, 0, -2, -1).clone(), eps);
        break;
      case PrecondKind::BlockJacobi:
      {
        _blk_inv.clear();
        _blk_offs.clear();
        int64_t o = 0;
        for (auto s : _block_sizes)
        {
          const int64_t i = o, j = o + s;
          auto blk = A.slice(-2, i, j).slice(-1, i, j); // (B, s, s)
          _blk_inv.push_back(std::get<0>(at::linalg_inv_ex(blk, /*check_errors=*/false)));
          _blk_offs.emplace_back(i, j);
          o = j;
        }
        break;
      }
      case PrecondKind::Full:
      {
        auto res = at::linalg_lu_factor_ex(A, /*pivot=*/true, /*check_errors=*/false);
        _lu = std::get<0>(res);
        _piv = std::get<1>(res);
        break;
      }
    }
    _ready = true;
  }

  /// Apply `M^-1` to `r` (`(B, N)`), returning `(B, N)`. Identity until `setup`.
  at::Tensor apply(const at::Tensor & r) const
  {
    switch (_kind)
    {
      case PrecondKind::None:
        return r;
      case PrecondKind::Jacobi:
        return _ready ? r / _diag : r;
      case PrecondKind::BlockJacobi:
      {
        if (!_ready)
          return r;
        auto out = at::empty_like(r);
        for (std::size_t k = 0; k < _blk_offs.size(); ++k)
        {
          const auto [i, j] = _blk_offs[k];
          out.slice(-1, i, j).copy_(at::einsum("bpq,bq->bp", {_blk_inv[k], r.slice(-1, i, j)}));
        }
        return out;
      }
      case PrecondKind::Full:
        return _ready ? at::linalg_lu_solve(_lu, _piv, r.unsqueeze(-1)).squeeze(-1) : r;
    }
    return r;
  }

private:
  PrecondKind _kind = PrecondKind::None;
  std::vector<int64_t> _block_sizes;
  at::Tensor _diag;                 // Jacobi
  std::vector<at::Tensor> _blk_inv; // BlockJacobi
  std::vector<std::pair<int64_t, int64_t>> _blk_offs;
  at::Tensor _lu, _piv; // Full
  bool _ready = false;
};
} // namespace neml2::aoti
