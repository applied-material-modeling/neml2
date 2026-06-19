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
#include <vector>

#include <ATen/core/Tensor.h>

#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/aoti/aoti_export.h"

namespace neml2::aoti
{
class NonlinearSystem;

// SolverConfig (the Newton tunables) is the public type declared in Model.h --
// it is also the argument to Model::set_solver_config.

/// Outcome of a *successful* Newton solve: the converged per-unknown-group
/// iterate and the iteration count. ``converged`` is always ``true`` here --
/// failure (divergence or hitting ``miters``) throws ``ConvergenceError`` rather
/// than returning, so a returned result is by construction converged.
struct NewtonResult
{
  std::vector<at::Tensor> u;
  bool converged = false;
  std::size_t iterations = 0;
};

/// Per-group Newton-Raphson solver with optional backtracking line search.
/// Convergence is the elementwise ``||b|| < atol OR ||b||/||b0|| < rtol``
/// all-reduce. Both failure modes -- divergence (non-finite residual) and
/// exhausting ``miters`` without converging -- throw ``ConvergenceError`` (a
/// *recoverable* error: a time-stepping consumer can cut its step and retry).
/// The all-reduce means a solve fails if *any* batch member is unconverged.
///
/// AOTI_EXPORT: this class is part of the shared library's public ABI so the
/// pybind layer (a separate module) can construct + drive it for the eager path.
class AOTI_EXPORT Newton
{
public:
  explicit Newton(SolverConfig cfg);

  /// Solve ``r(u) = 0`` starting from ``u0`` (per unknown group). Set
  /// ``NEML2_AOTI_TRACE_NEWTON=1`` (or ``2`` for per-iteration detail) to trace
  /// to stderr.
  NewtonResult solve(const NonlinearSystem & sys, const std::vector<at::Tensor> & u0) const;

private:
  SolverConfig _cfg;
};
} // namespace neml2::aoti
