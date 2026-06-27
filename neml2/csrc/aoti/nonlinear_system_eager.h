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
// delegates residual/step to Python callables (typically RHS / NewtonStep from
// neml2.es.implicit), so the eager and AOTI paths run the *same* C++ iteration.

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <pybind11/pybind11.h>

#include "neml2/csrc/aoti/newton.h"

namespace neml2::aoti
{
/// Drive the shared C++ Newton solver over an eager (Python-delegating) system.
///
/// ``residual_fn(list[Tensor] u) -> list[Tensor] b`` and
/// ``step_fn(list[Tensor] u) -> (list[Tensor] du, list[Tensor] b)`` are Python
/// callables that already bind the givens + linear solver (e.g. RHS /
/// NewtonStep). ``unknown_layout`` / ``residual_layout`` are ``(structure,
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
} // namespace neml2::aoti
