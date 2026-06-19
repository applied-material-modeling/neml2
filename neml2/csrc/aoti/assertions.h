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
///   - a foreign exception (a torch `c10::Error` from a shape / device mismatch
///     inside a compiled graph, `std::bad_alloc`, ...) becomes a `FatalError`,
///     i.e. non-recoverable -- a caller that retries on `recoverable()` will
///     never retry one of these.
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
  catch (const std::exception & e)
  {
    throw FatalError(std::string("aoti: ") + e.what());
  }
  catch (...)
  {
    throw FatalError("aoti: unknown (non-std) exception");
  }
}
} // namespace neml2::aoti
