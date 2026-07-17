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

// SHIPPED header -- part of the public C++ ABI. The exception taxonomy a
// downstream consumer catches off `Model` / `DispatchedModel`.
//
// The distinction that matters to a consumer (e.g. a finite-element solver with
// time stepping) is **recoverable vs. not**: a numerical failure such as a
// nonlinear-solve non-convergence can often be cleared by cutting the time step
// and retrying, whereas a shape / device / configuration error cannot -- a retry
// would just fail the same way, so it must hard-fail. Branch on
// `Exception::recoverable()`:
//
//   try { out = model.forward(in); }
//   catch (const neml2::aoti::Exception & e)
//   {
//     if (e.recoverable()) { dt *= 0.5; continue; }   // e.g. cut + retry
//     throw;                                           // fatal: give up
//   }
//
// Every exception the runtime throws derives from `Exception` (itself a
// `std::runtime_error`), so existing `catch (const std::exception &)` sites keep
// working; only code that wants the recoverable/fatal split needs the new types.

#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/core/Tensor.h>

#include "neml2/csrc/aoti/aoti_export.h"

namespace neml2::aoti
{
/// Base for every exception the neml2 runtime throws. Carries a `recoverable()`
/// flag so a downstream consumer can decide whether to retry (e.g. cut the time
/// step) or hard-fail. Not thrown directly -- the concrete leaves below are.
class AOTI_EXPORT Exception : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;

  /// Whether the caller may sensibly retry with adjusted inputs. `false` for
  /// programming / configuration errors (shape, device, malformed artifact);
  /// `true` for numerical failures such as nonlinear-solve non-convergence.
  virtual bool recoverable() const noexcept;
};

/// A non-recoverable error: a shape / device mismatch, a missing input, a
/// malformed artifact -- anything a retry cannot fix. The type the runtime's
/// internal assertions throw.
class AOTI_EXPORT FatalError : public Exception
{
public:
  using Exception::Exception;
  bool recoverable() const noexcept override;
};

/// A recoverable error: the computation failed numerically but the caller may
/// retry with adjusted inputs. The canonical case is a nonlinear solve that
/// diverged or hit its iteration cap -- a time-stepping consumer can cut the
/// step and try again.
class AOTI_EXPORT ConvergenceError : public Exception
{
public:
  using Exception::Exception;

  /// Enriched constructor that additionally carries a *failure context* for
  /// offline debugging: a per-dynamic-batch-element convergence mask (`true`
  /// where the element converged) and the best-effort iterate the solve got
  /// stuck at, keyed by unknown-variable name. This context is populated only on
  /// the opt-in capture path (the `NEML2_CAPTURE_SOLVE_FAILURE` env var); the
  /// message-only constructor above leaves both empty. The pybind layer surfaces
  /// them to Python as the `converged_mask` / `unknown` attributes so an offline
  /// replay can read the stuck state with near-zero boilerplate.
  ConvergenceError(const std::string & what_arg,
                   at::Tensor converged_mask,
                   std::map<std::string, at::Tensor> unknowns);

  bool recoverable() const noexcept override;

  /// Per-dynamic-batch-element convergence mask captured at failure (an undefined
  /// tensor when capture was disabled). `true` where the element converged.
  const at::Tensor & converged_mask() const noexcept { return _converged_mask; }

  /// Best-effort iterate at failure, keyed by unknown-variable name (empty when
  /// capture was disabled).
  const std::map<std::string, at::Tensor> & unknowns() const noexcept { return _unknowns; }

private:
  at::Tensor _converged_mask;
  std::map<std::string, at::Tensor> _unknowns;
};

/// Several concurrent dispatches failed at once (the asynchronous
/// @ref DispatchedModel path). Holds every sub-error, in dispatch order; its
/// `recoverable()` is `true` only if **every** sub-error is recoverable -- a
/// single fatal makes the whole aggregate fatal, since a retry cannot fix a
/// shape / device bug even if the other chunks merely failed to converge.
class AOTI_EXPORT AggregateError : public Exception
{
public:
  /// @param errors the individual chunk failures (must be non-empty). Each is a
  ///        captured `std::exception_ptr`; the message + recoverability are
  ///        derived from them at construction.
  explicit AggregateError(std::vector<std::exception_ptr> errors);

  bool recoverable() const noexcept override;

  /// The individual chunk failures, in dispatch order, for inspection /
  /// re-throw by the caller.
  const std::vector<std::exception_ptr> & errors() const noexcept { return _errors; }

private:
  std::vector<std::exception_ptr> _errors;
  bool _recoverable;
};
} // namespace neml2::aoti
