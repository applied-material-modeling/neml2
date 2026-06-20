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

// NON-shipped internal header (kept out of the wheel via wheel.exclude and out
// of the installed FILE_SET HEADERS). Python-free by design so it can sit in a
// member of Model::Impl without leaking pybind into the public surface.

namespace neml2::eager
{
/**
 * @brief Ensures an embedded CPython interpreter is running for this process.
 *
 * Declared as the *first* member of @ref Model::Impl so the interpreter is up
 * (construction order = declaration order) before the pybind adapter object is
 * imported and created.
 *
 * Behavior is mode-aware:
 * - **Pure-C++ host:** the first guard calls `pybind11::initialize_interpreter()`
 *   and releases the GIL for the process lifetime, so subsequent
 *   `gil_scoped_acquire` calls (including from worker threads) work.
 * - **Host already running Python:** the guard detects the live interpreter and
 *   leaves its lifecycle untouched.
 *
 * The interpreter is intentionally **never finalized** (see interpreter.cpp for
 * why); it lives until process exit. The destructor is therefore a no-op.
 */
class InterpreterGuard
{
public:
  InterpreterGuard();
  ~InterpreterGuard();
  InterpreterGuard(const InterpreterGuard &) = delete;
  InterpreterGuard & operator=(const InterpreterGuard &) = delete;
  InterpreterGuard(InterpreterGuard &&) = delete;
  InterpreterGuard & operator=(InterpreterGuard &&) = delete;
};
} // namespace neml2::eager
