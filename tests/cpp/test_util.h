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

// Minimal, dependency-free test scaffolding for the aoti C++ tests. Each test
// is a plain `main()` that returns non-zero on the first failed check -- no
// third-party framework. If the test matrix grows enough to need fixtures /
// rich reporting, swap this for a real framework then.

#include <cstdio>

// Namespaced so they don't collide with torch's own CHECK macro (the glog shim
// in c10/util/logging_is_not_google_glog.h defines CHECK).

// Fail the enclosing `main()` (return 1) if `cond` is false.
#define NEML2_CHECK(cond)                                                                          \
  do                                                                                               \
  {                                                                                                \
    if (!(cond))                                                                                   \
    {                                                                                              \
      std::fprintf(stderr, "NEML2_CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__);     \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

// Fail the enclosing `main()` unless evaluating `expr` throws.
#define NEML2_CHECK_THROWS(expr)                                                                   \
  do                                                                                               \
  {                                                                                                \
    bool _threw = false;                                                                           \
    try                                                                                            \
    {                                                                                              \
      (void)(expr);                                                                                \
    }                                                                                              \
    catch (...)                                                                                    \
    {                                                                                              \
      _threw = true;                                                                               \
    }                                                                                              \
    if (!_threw)                                                                                   \
    {                                                                                              \
      std::fprintf(stderr,                                                                         \
                   "NEML2_CHECK_THROWS failed (no throw): %s\n  at %s:%d\n",                       \
                   #expr,                                                                          \
                   __FILE__,                                                                       \
                   __LINE__);                                                                      \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)
