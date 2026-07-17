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

// Lives in the pybind module (NOT the free-standing aoti library): bridges the
// shared C++ Newton solver to the eager Python path. `EagerNonlinearSystem`
// delegates residual/step to Python callables (typically RHS / (Jacobian -> LinearSolve) from
// neml2.es.implicit), so the eager and AOTI paths run the *same* C++ iteration.

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <pybind11/pybind11.h>

#include "neml2/csrc/aoti/krylov.h"
#include "neml2/csrc/aoti/newton.h"

namespace neml2::aoti
{
/// Drive the shared C++ Newton solver over an eager (Python-delegating) system.
///
/// ``residual_fn(list[Tensor] u) -> list[Tensor] b`` and
/// ``step_fn(list[Tensor] u) -> (list[Tensor] du, list[Tensor] b)`` are Python
/// callables that already bind the givens + linear solver (e.g. RHS +
/// Jacobian->LinearSolve). ``unknown_layout`` / ``residual_layout`` are ``(structure,
/// sub_batch_shape)`` per group. Returns ``(u_solved, converged, iterations,
/// log)`` where ``log`` is the per-iteration convergence history (empty unless
/// ``cfg.collect_log`` is set). Must be called with the GIL held (it invokes
/// the Python callables inline).
std::tuple<std::vector<at::Tensor>, bool, std::size_t, std::vector<std::string>>
run_eager_newton(const SolverConfig & cfg,
                 pybind11::object residual_fn,
                 pybind11::object step_fn,
                 std::vector<std::pair<std::string, std::vector<int64_t>>> unknown_layout,
                 std::vector<std::pair<std::string, std::vector<int64_t>>> residual_layout,
                 std::vector<at::Tensor> u0);

/// Masking variant of run_eager_newton: drives ``Newton::solve_masked`` (which
/// never throws) and returns ``(u_best, converged_mask, iterations)`` -- the
/// best-effort iterate, the per-dynamic-batch-element convergence mask (``true``
/// where the element converged), and the iteration count. Used to capture a
/// failed batch's non-converged state and which elements diverged, for offline
/// debugging (see ``NEML2_DUMP_SOLVE_FAILURE``). Must be called with the GIL held.
std::tuple<std::vector<at::Tensor>, at::Tensor, std::size_t>
run_eager_newton_masked(const SolverConfig & cfg,
                        pybind11::object residual_fn,
                        pybind11::object step_fn,
                        std::vector<std::pair<std::string, std::vector<int64_t>>> unknown_layout,
                        std::vector<std::pair<std::string, std::vector<int64_t>>> residual_layout,
                        std::vector<at::Tensor> u0);

/// Drive the shared C++ Newton solver over an eager system whose inner linear
/// solve is a matrix-free Krylov iteration (krylov.h) rather than a direct solve.
///
/// ``residual_fn(list[Tensor] u) -> list[Tensor] b`` and
/// ``matvec_fn(list[Tensor] u, list[Tensor] v) -> list[Tensor] Jv`` supply the
/// residual and the matrix-free `J.v` (the eager RHS / Matvec modules).
/// ``precond_setup_fn(list[Tensor] u) -> list[Tensor] state`` and
/// ``precond_apply_fn(list[Tensor] state, Tensor r_flat) -> Tensor z_flat`` are
/// the authored preconditioner's setup/apply modules (both ``None`` when
/// unpreconditioned). Same return tuple as ``run_eager_newton``. Must be called
/// with the GIL held.
std::tuple<std::vector<at::Tensor>, bool, std::size_t, std::vector<std::string>>
run_eager_krylov(const SolverConfig & newton_cfg,
                 const KrylovConfig & krylov_cfg,
                 pybind11::object residual_fn,
                 pybind11::object matvec_fn,
                 pybind11::object precond_setup_fn,
                 pybind11::object precond_apply_fn,
                 std::vector<std::pair<std::string, std::vector<int64_t>>> unknown_layout,
                 std::vector<std::pair<std::string, std::vector<int64_t>>> residual_layout,
                 std::vector<at::Tensor> u0);
} // namespace neml2::aoti
