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

#include "neml2/csrc/eager/interpreter.h"

#include <mutex>

#include <pybind11/embed.h>
#include <pybind11/gil.h>

namespace py = pybind11;

namespace neml2::eager
{
namespace
{
// Process-wide bootstrap state, guarded by a mutex so concurrent first-time
// `eager::Model` constructions don't race to initialize the interpreter.
std::mutex g_mutex;
bool g_started_by_us = false;
// The GIL is released for the interpreter's lifetime via this sentinel so that
// later `gil_scoped_acquire` calls -- including from other host threads -- can
// acquire it. `pybind11::initialize_interpreter()` returns with the GIL held;
// this sentinel hands it back. Intentionally leaked (never deleted): we never
// finalize the interpreter, so the GIL stays released until process exit.
py::gil_scoped_release * g_release = nullptr;
} // namespace

InterpreterGuard::InterpreterGuard()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  // Already running -- either we started it on an earlier construction, or the
  // host process owns the interpreter (we were imported under Python). Either
  // way, do nothing: never double-initialize, never touch a foreign lifecycle.
  if (g_started_by_us || Py_IsInitialized())
    return;
  py::initialize_interpreter();
  // Line-buffer the interpreter's stdio. We never finalize the interpreter (see
  // the destructor), so Python's atexit flush never runs; a block-buffered
  // stream -- the default when the host's stdout is not a TTY (redirected to a
  // file or pipe) -- would never be flushed and its output would be lost. Line
  // buffering flushes on each newline, so Python-side prints (e.g. a solver's
  // `verbose` convergence log) surface whether or not stdout is a TTY.
  {
    auto sys = py::module_::import("sys");
    for (const char * name : {"stdout", "stderr"})
    {
      py::object stream = sys.attr(name);
      if (!stream.is_none() && py::hasattr(stream, "reconfigure"))
        stream.attr("reconfigure")(py::arg("line_buffering") = true);
    }
  }
  g_release = new py::gil_scoped_release();
  g_started_by_us = true;
}

// Intentionally a no-op: we never call `pybind11::finalize_interpreter()`.
//
// CPython + PyTorch do not support cleanly re-importing `torch` after
// `Py_FinalizeEx`, so a finalize-then-reinitialize cycle (e.g. a C++ test that
// constructs several eager models in sequence, each fully destroyed before the
// next) would crash inside torch's re-init. Leaving the interpreter up until
// process exit -- where the OS reclaims it -- sidesteps that entirely and also
// removes the finalize-vs-py::object-teardown ordering hazard. When the host
// already owned the interpreter we likewise must not finalize it.
InterpreterGuard::~InterpreterGuard() = default;
} // namespace neml2::eager
