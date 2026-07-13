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

// Standalone numerics test for the batched matrix-free Krylov solvers
// (krylov.h). Uses an in-memory dense operator `matvec(v) = A @ v` (no AOTI
// loaders, no NonlinearSystem), so it validates the ported GMRES / BiCGStab +
// the functor interface in isolation against `at::linalg_solve`.

#include <cstdio>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/krylov.h"
#include "neml2/csrc/aoti/krylov_flatten.h"

#include "test_util.h"

using namespace neml2::aoti;

namespace
{
// A well-conditioned (diagonally dominant, near-identity) batched operator --
// the favorable regime for GMRES, mirroring the implicit backward-Euler
// Jacobians whose spectrum clusters near 1.
at::Tensor
make_operator(int64_t B, int64_t n, const at::TensorOptions & opts)
{
  return 0.02 * at::randn({B, n, n}, opts) + at::eye(n, opts);
}

at::Tensor
apply_dense(const at::Tensor & A, const at::Tensor & v) // (B,n,n),(B,n) -> (B,n)
{
  return at::einsum("bnm,bm->bn", {A, v});
}

double
rel_residual(const at::Tensor & A, const at::Tensor & x, const at::Tensor & b)
{
  auto num = (apply_dense(A, x) - b).norm(2, -1);
  auto den = b.norm(2, -1).clamp_min(1e-30);
  return (num / den).max().item<double>();
}

// Solve A x = b with the given method + preconditioner functor and check the
// residual + agreement with the direct solve. Returns 0 on success, 1 on failure.
int
check_solve(at::ScalarType dtype,
            KrylovMethod method,
            const PrecondFn & minv,
            double resid_bar,
            double direct_bar)
{
  at::manual_seed(20240712);
  const auto opts = at::TensorOptions().dtype(dtype);
  const int64_t Bsz = 8, n = 20;
  auto A = make_operator(Bsz, n, opts);
  auto b = at::randn({Bsz, n}, opts);

  const auto matvec = [&A](const at::Tensor & v) { return apply_dense(A, v); };

  KrylovConfig cfg;
  cfg.method = method;
  cfg.restart = n;       // full restart width -> converges within a cycle
  cfg.max_iters = 8 * n; // a few restarts of headroom
  // A strict-but-achievable inner tolerance: GMRES(m=n) reaches ~machine eps in
  // one cycle, but BiCGStab stagnates near ~1e-10 on this problem (as it does in
  // general) -- 1e-11 would be unreachable for it. Production inner-Krylov tols
  // are far looser (inexact Newton), so this only stresses the numerics.
  cfg.rel_tol = (dtype == at::kFloat) ? 1e-5 : 1e-9;
  cfg.abs_tol = 0.0;

  auto res = krylov_solve(matvec, minv, b, cfg);

  // Every batch element must have converged within the budget.
  NEML2_CHECK(res.converged.all().item<bool>());

  // Residual b - A x is small.
  const double rr = rel_residual(A, res.du, b);
  if (!(rr < resid_bar))
  {
    std::fprintf(stderr,
                 "  krylov residual %.3e >= bar %.3e (dtype=%d method=%d)\n",
                 rr,
                 resid_bar,
                 static_cast<int>(dtype),
                 static_cast<int>(method));
    return 1;
  }

  // Agreement with the direct solve.
  auto x_direct = at::linalg_solve(A, b.unsqueeze(-1)).squeeze(-1);
  const double derr = ((res.du - x_direct).norm(2, -1) / x_direct.norm(2, -1).clamp_min(1e-30))
                          .max()
                          .item<double>();
  if (!(derr < direct_bar))
  {
    std::fprintf(stderr,
                 "  krylov vs direct %.3e >= bar %.3e (dtype=%d method=%d)\n",
                 derr,
                 direct_bar,
                 static_cast<int>(dtype),
                 static_cast<int>(method));
    return 1;
  }
  return 0;
}

// Build the Preconditioner of the given kind from a dense A and check that (a)
// its apply drives GMRES to the direct solution, and (b) for Full it makes GMRES
// converge in a single iteration (M^-1 == A^-1). Returns 0 on success.
int
check_precond(PrecondKind kind, const std::vector<int64_t> & blocks)
{
  at::manual_seed(1234);
  const auto opts = at::TensorOptions().dtype(at::kDouble);
  const int64_t Bsz = 6, n = 20;
  auto A = make_operator(Bsz, n, opts);
  auto b = at::randn({Bsz, n}, opts);

  Preconditioner pc(kind, blocks);
  NEML2_CHECK(pc.kind() != PrecondKind::None ? !pc.ready() : pc.ready());
  pc.setup(A);
  NEML2_CHECK(pc.ready());

  const auto matvec = [&A](const at::Tensor & v) { return apply_dense(A, v); };
  const PrecondFn minv = [&pc](const at::Tensor & r) { return pc.apply(r); };

  KrylovConfig cfg;
  cfg.restart = n;
  cfg.max_iters = 8 * n;
  cfg.rel_tol = 1e-9;
  auto res = krylov_solve(matvec, minv, b, cfg);
  NEML2_CHECK(res.converged.all().item<bool>());

  const double rr = rel_residual(A, res.du, b);
  NEML2_CHECK(rr < 1e-8);

  // A full-Jacobian preconditioner is exact -> GMRES converges in one iteration.
  if (kind == PrecondKind::Full)
    NEML2_CHECK(res.max_iters <= 1);
  return 0;
}

// flatten_dense / unflatten_dense roundtrip over a multi-dim dynamic batch.
int
check_flatten()
{
  const auto opts = at::TensorOptions().dtype(at::kDouble);
  std::vector<GroupLayout> layout = {GroupLayout{"dense", {}}, GroupLayout{"dense", {}}};
  auto g0 = at::randn({4, 5, 3}, opts); // (*B=(4,5), w=3)
  auto g1 = at::randn({4, 5, 7}, opts); // (*B=(4,5), w=7)
  std::vector<at::Tensor> groups = {g0, g1};

  FlatSpec spec;
  auto flat = flatten_dense(groups, layout, spec);
  NEML2_CHECK(flat.dim() == 2 && flat.size(0) == 20 && flat.size(1) == 10);
  NEML2_CHECK(spec.N == 10 && spec.widths.size() == 2 && spec.widths[0] == 3);

  auto back = unflatten_dense(flat, spec);
  NEML2_CHECK(back.size() == 2);
  NEML2_CHECK((back[0] - g0).abs().max().item<double>() == 0.0);
  NEML2_CHECK((back[1] - g1).abs().max().item<double>() == 0.0);

  // A BLOCK group must be rejected up front.
  std::vector<GroupLayout> block_layout = {GroupLayout{"block", {12}}};
  NEML2_CHECK_THROWS(assert_all_dense(block_layout, "unknown"));
  return 0;
}
} // namespace

int
main()
{
  const PrecondFn identity = [](const at::Tensor & v) { return v; };

  // GMRES + BiCGStab, no preconditioner, double + float.
  NEML2_CHECK(check_solve(at::kDouble, KrylovMethod::GMRES, identity, 1e-8, 1e-6) == 0);
  NEML2_CHECK(check_solve(at::kDouble, KrylovMethod::BiCGStab, identity, 1e-8, 1e-6) == 0);
  NEML2_CHECK(check_solve(at::kFloat, KrylovMethod::GMRES, identity, 1e-3, 1e-2) == 0);
  NEML2_CHECK(check_solve(at::kFloat, KrylovMethod::BiCGStab, identity, 1e-3, 1e-2) == 0);

  // A preconditioner functor (Jacobi: divide by the operator diagonal) must also
  // drive both methods to the same solution -- exercises the PrecondFn path.
  {
    at::manual_seed(20240712);
    const auto opts = at::TensorOptions().dtype(at::kDouble);
    auto A = make_operator(8, 20, opts);
    auto diag = at::diagonal(A, 0, -2, -1).clone(); // (B, n)
    const PrecondFn jacobi = [diag](const at::Tensor & v) { return v / diag; };
    NEML2_CHECK(check_solve(at::kDouble, KrylovMethod::GMRES, jacobi, 1e-8, 1e-6) == 0);
    NEML2_CHECK(check_solve(at::kDouble, KrylovMethod::BiCGStab, jacobi, 1e-8, 1e-6) == 0);
  }

  // The Preconditioner class (setup from a dense A + cached apply), each kind.
  NEML2_CHECK(check_precond(PrecondKind::None, {}) == 0);
  NEML2_CHECK(check_precond(PrecondKind::Jacobi, {}) == 0);
  NEML2_CHECK(check_precond(PrecondKind::BlockJacobi, {5, 5, 5, 5}) == 0);
  NEML2_CHECK(check_precond(PrecondKind::Full, {}) == 0);

  // Cross-group flatten/unflatten roundtrip + BLOCK rejection.
  NEML2_CHECK(check_flatten() == 0);

  std::printf("test_krylov: all checks passed\n");
  return 0;
}
