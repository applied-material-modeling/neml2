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

#include <sstream>

// c10::LinAlgError is a numerical failure raised by torch's linalg kernels
// (a singular / non-invertible / non-PD factorization); a downstream time-
// stepping consumer can recover from it by cutting the step, so the guard below
// re-throws it as the recoverable neml2 ConvergenceError rather than letting it
// fall through to the generic std::exception catch (which would flag it fatal).
#include <c10/util/Exception.h>

#include "neml2/csrc/aoti/Exception.h"

// Internal header -- NOT shipped. Holds the check-and-throw helpers shared
// across the aoti runtime's translation units.
//
// These are deliberately kept here (rather than pulling in the legacy
// `neml2/misc/assertions.h`) so the AOTI submodule stays free-standing: it
// links only against torch::core + nlohmann_json and can outlive the rest of
// the C++ object tower. They are function templates, so a header definition is
// ODR-safe across the TUs that include it. The streaming-args ergonomics match
// the `neml_assert` style at the call sites.
//
// `_assert` / `_throw` raise a `FatalError`: an assertion firing means a
// shape / device / configuration / metadata invariant was violated, which a
// retry cannot fix. Numerical failures that a consumer *can* recover from (e.g.
// nonlinear-solve non-convergence) throw `ConvergenceError` at their own sites
// instead -- never through `_assert`.
namespace neml2::aoti
{
template <typename... Args>
[[noreturn]] inline void
_throw(const Args &... args)
{
  std::ostringstream oss;
  ((oss << args), ...);
  throw FatalError(oss.str());
}

template <typename... Args>
inline void
_assert(bool cond, const Args &... args)
{
  if (!cond)
    _throw(args...);
}

/// Run @p fn at a public API boundary, normalizing whatever escapes into the
/// neml2 exception taxonomy so a consumer has a single catch surface:
///   - a neml2 `Exception` (incl. the recoverable `ConvergenceError`) passes
///     through untouched -- its dynamic type and `recoverable()` are preserved;
///   - a torch `c10::LinAlgError` (a singular / non-invertible / non-PD linear
///     factorization from a torch linalg kernel) is re-thrown as the recoverable
///     `ConvergenceError` -- like a Newton non-convergence, a time-stepping
///     consumer can cut the step and retry. Recognized BOTH as the typed C++
///     exception (when RTTI matches across libraries) AND by the preserved
///     "torch.linalg.<op>:" message prefix on a plain `std::exception` (torch's
///     AOTI proxy_executor C API strips derived exception types when it re-
///     throws across the C boundary, so a c10::LinAlgError from inside an AOTI-
///     compiled graph reaches the caller as c10::Error/std::exception -- the
///     original message text survives even though the derived type does not);
///   - any other foreign exception (a torch `c10::Error` from a shape / device
///     mismatch inside a compiled graph, `std::bad_alloc`, ...) becomes a
///     `FatalError`, i.e. non-recoverable -- a caller that retries on
///     `recoverable()` will never retry one of these.
template <typename Fn>
auto
_guarded(Fn && fn) -> decltype(fn())
{
  try
  {
    return fn();
  }
  catch (const Exception &)
  {
    throw;
  }
  catch (const c10::LinAlgError & e)
  {
    // Fast path: RTTI matched. Direct-throw sites (or same-library callers)
    // land here. Uses `what()` to include the torch backtrace, matching the
    // FatalError catch below.
    throw ConvergenceError(std::string("aoti: torch linalg failure: ") + e.what());
  }
  catch (const std::exception & e)
  {
    const std::string what = e.what();
    // Fallback for a c10::LinAlgError that reached here without matching the
    // typed catch above: torch's AOTI proxy_executor C API (see aoti_torch_
    // proxy_executor_call_function in libtorch) catches every std::exception
    // inside the compiled kernel and re-throws on the C++ side as a fresh
    // c10::Error carrying the original what() text but not the derived type.
    // A hidden-visibility RTTI split across libraries has the same effect.
    // Detect via the "torch.linalg.<op>:" prefix torch always puts at the head
    // of a real linalg failure's message; that prefix survives both cases.
    if (what.find("torch.linalg.") != std::string::npos)
      throw ConvergenceError(std::string("aoti: torch linalg failure: ") + what);
    throw FatalError(std::string("aoti: ") + what);
  }
  catch (...)
  {
    throw FatalError("aoti: unknown (non-std) exception");
  }
}
} // namespace neml2::aoti
