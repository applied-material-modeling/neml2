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

// Internal header -- NOT shipped. The model-agnostic, per-group Newton solver.
// It drives an abstract `NonlinearSystem` and knows nothing about AOTI, the
// `Segment` metadata, or where the residual/step come from -- that is the whole
// point: one iteration-control implementation shared by the compiled (AOTI) and
// (eventually) eager paths.

#include <cstddef>
#include <string>
#include <vector>

#include <ATen/core/Tensor.h>

namespace neml2::aoti
{
class NonlinearSystem;

/// Tunables for the Newton iteration. Mirrors the HIT `Newton` /
/// `NewtonWithLineSearch` options; line search is enabled iff
/// ``ls_max_iters > 1``. ``ls_type`` is "BACKTRACKING" or "STRONG_WOLFE".
struct SolverConfig
{
  double atol = 1.0e-10;
  double rtol = 1.0e-8;
  std::size_t miters = 25;
  std::string ls_type = "BACKTRACKING";
  std::size_t ls_max_iters = 1;
  double ls_cutback = 2.0;
  double ls_c = 1.0e-3;
};

/// Per-group Newton-Raphson solver with optional backtracking line search.
/// Convergence is the elementwise ``||b|| < atol OR ||b||/||b0|| < rtol``
/// all-reduce; divergence (non-finite residual) throws. On max-iterations it
/// returns the last iterate (matching the NEML2 Newton convention).
class Newton
{
public:
  explicit Newton(SolverConfig cfg);

  /// Solve ``r(u) = 0`` starting from ``u0`` (per unknown group). Returns the
  /// converged per-unknown-group vector. Set ``NEML2_AOTI_TRACE_NEWTON=1`` (or
  /// ``2`` for per-iteration detail) to trace to stderr.
  std::vector<at::Tensor> solve(const NonlinearSystem & sys,
                                const std::vector<at::Tensor> & u0) const;

private:
  SolverConfig _cfg;
};
} // namespace neml2::aoti
