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

// SHIPPED header -- part of the public C++ ABI. The central verbosity / logging
// configuration shared by every NEML2 evaluation route.
//
// This is the single source of truth for "how loud is neml2, and where does its
// output go". The same store backs the C++ solvers (Newton / Krylov / substep)
// *and* the Python layer (via the `_aoti` pybind binding + `neml2.log`), so a
// model evaluated through any of the six routes obeys one configuration.
//
// Two layers, resolved highest-first:
//   1. the `NEML2_LOGS` env var (end-user override, e.g. `newton=info,substep=debug`)
//   2. programmatic defaults set by a downstream host (`set_default_level`)
//   3. the built-in default (`Level::Warning`)
// so a host app can pick its own defaults while its end-users still override via
// the environment. Within a layer, an explicit channel beats the `all` baseline.
//
// Output routing is a single settable `Sink`. The default sink splits by level:
// `Info` -> stdout, `Warning` / `Debug` -> stderr, one flushed line each. A
// downstream host (e.g. a finite-element code) can `set_sink` to route neml2's
// log lines into its own console instead.

#include <functional>
#include <string>

#include "neml2/csrc/aoti/aoti_export.h"

namespace neml2::aoti::log
{
/// Verbosity levels, ordered least-to-most verbose. A message emitted at level
/// `L` is shown when a channel's effective level is `>= L`.
enum class Level
{
  Silent = 0,  ///< nothing
  Warning = 1, ///< advisories (predictor fallback, degraded config)
  Info = 2,    ///< per-solve summaries, "results saved", solve banners
  Debug = 3,   ///< per-iteration / per-span detail, metadata dumps
};

/// The diagnostic channels. Component-oriented, not feature-specific, so new
/// diagnostics slot into an existing channel rather than minting a new knob.
enum class Channel
{
  Newton = 0,  ///< nonlinear (Newton) solve: summary + banners (info), iterations (debug)
  Linear = 1,  ///< inner linear-solve (Krylov) residual history (debug)
  Substep = 2, ///< adaptive sub-incrementation: summary (info), per-span (debug)
  Model = 3,   ///< model / equation-system evaluation + framework advisories
  Tensor = 4,  ///< typed-tensor-level diagnostics
  Driver = 5,  ///< drivers: "results saved", per-step notices
};

/// A sink receives the level and the fully-formatted line (already carrying the
/// `[neml2:<channel>] ` prefix). Install one with @ref set_sink to redirect all
/// neml2 output; the level lets a sink route by severity if it wishes.
using Sink = std::function<void(Level, const std::string &)>;

/// Whether `channel` would emit a message at `level` (effective level >= level).
/// Cheap; call it to guard the (possibly expensive) construction of a log line.
AOTI_EXPORT bool enabled(Channel channel, Level level);

/// The effective level of `channel` after applying env > programmatic > default.
AOTI_EXPORT Level effective_level(Channel channel);

/// Format then emit `message` on `channel` at `level` iff enabled. The line is
/// `[neml2:<channel>] <message>`.
AOTI_EXPORT void emit(Channel channel, Level level, const std::string & message);

/// The formatted line (`[neml2:<channel>] <message>`) without gating or emitting.
AOTI_EXPORT std::string format(Channel channel, const std::string & message);

/// Emit a begin/end solve banner (`---- begin <label> ----`) at `Info`, so a
/// downstream consumer embedding neml2 solves inside its own residual/Jacobian
/// evaluations can bracket neml2's output.
AOTI_EXPORT void begin_solve(Channel channel, const std::string & label);
AOTI_EXPORT void end_solve(Channel channel, const std::string & label);

/// Programmatic default level for one channel / for every channel. Set by a
/// downstream host to pick its baseline verbosity. An `NEML2_LOGS` entry for the
/// same channel still overrides it.
AOTI_EXPORT void set_default_level(Channel channel, Level level);
AOTI_EXPORT void set_default_level(Level level);
/// Clear all programmatic defaults (revert to env > built-in default).
AOTI_EXPORT void reset_defaults();

/// Install a custom sink (e.g. to route into a host console). Passing an empty
/// `Sink` is treated as @ref reset_sink.
AOTI_EXPORT void set_sink(Sink sink);
/// Restore the default level-splitting stdout/stderr sink.
AOTI_EXPORT void reset_sink();

/// (Re)parse the `NEML2_LOGS` env var into the env layer. Applied lazily on first
/// use; re-callable so a test can set the variable and refresh.
AOTI_EXPORT void apply_env();

/// Canonical lowercase name of a channel (`"newton"`, ...). Used by the prefix
/// and the pybind bridge.
AOTI_EXPORT const char * channel_name(Channel channel);
/// Canonical lowercase name of a level (`"silent"`, `"warning"`, ...). Used by
/// the pybind bridge to hand a Python sink the severity as a string.
AOTI_EXPORT const char * level_name(Level level);
/// Parse a channel / level name (case-insensitive). Returns false on an unknown
/// token so the caller can skip it. `"all"` is NOT a channel here -- it is handled
/// by the env parser / @ref set_default_level(Level).
AOTI_EXPORT bool parse_channel(const std::string & s, Channel & out);
AOTI_EXPORT bool parse_level(const std::string & s, Level & out);
} // namespace neml2::aoti::log
