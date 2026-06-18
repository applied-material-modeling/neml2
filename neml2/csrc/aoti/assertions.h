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
#include <stdexcept>

// Internal header -- NOT shipped. Holds the check-and-throw helpers shared
// across the aoti runtime's translation units.
//
// These are deliberately kept here (rather than pulling in the legacy
// `neml2/misc/assertions.h`) so the AOTI submodule stays free-standing: it
// links only against torch::core + nlohmann_json and can outlive the rest of
// the C++ object tower. They are function templates, so a header definition is
// ODR-safe across the TUs that include it. The streaming-args ergonomics match
// the `neml_assert` style at the call sites.
namespace neml2::aoti
{
template <typename... Args>
[[noreturn]] inline void
_throw(const Args &... args)
{
  std::ostringstream oss;
  ((oss << args), ...);
  throw std::runtime_error(oss.str());
}

template <typename... Args>
inline void
_assert(bool cond, const Args &... args)
{
  if (!cond)
    _throw(args...);
}
} // namespace neml2::aoti
