(verbosity-control)=
# Controlling verbosity

NEML2's diagnostic output — nonlinear-solver convergence traces, adaptive
substepping summaries, model / equation-system debug dumps, driver notices — is
governed by one configuration store, shared by every [evaluation
route](deployment-overview). The same store lives in the C++ runtime
(`libneml2`) and is surfaced to Python, so a model obeys the same settings
whether it runs eager in Python, as a compiled artifact in a C++ host, or split
across devices by the dispatcher.

There are two tiers, kept deliberately separate:

- **Library diagnostics** — what the solvers and drivers emit *during
  evaluation*. Controlled by the `NEML2_LOGS` environment variable (below),
  uniformly across all routes.
- **CLI chrome** — a command's *own* progress/status output (for example
  `neml2-compile`'s per-file progress). Not governed by `NEML2_LOGS`; each such
  command has its own `-q/--quiet` flag. A command's primary result and its
  errors are never suppressed.

## The `NEML2_LOGS` environment variable

`NEML2_LOGS` is a comma-separated list of `channel=level` entries, in the spirit
of PyTorch's `TORCH_LOGS`:

```bash
export NEML2_LOGS="newton=info,substep=debug"   # per-channel
export NEML2_LOGS="all=warning"                  # one baseline for every channel
export NEML2_LOGS="all=info,newton=silent"       # baseline + an override
```

**Levels**, least to most verbose: `silent` < `warning` < `info` < `debug`
(the integers `0`–`3` and the aliases `off`/`warn` are also accepted). As a rule
of thumb: `warning` for advisories, `info` for per-solve summaries and progress,
`debug` for per-iteration / per-item detail. A message emitted at level *L* is
shown when its channel's effective level is *L* or higher; `silent` mutes the
channel.

**Channels** name the subsystem a diagnostic comes from:

- `newton` — the nonlinear (Newton) solver
- `linear` — the inner linear (Krylov) solver
- `substep` — adaptive sub-incrementation
- `model` — model / equation-system evaluation
- `tensor` — typed-tensor operations
- `driver` — the drivers (running a model over a load history)

`all` is a wildcard that sets a baseline for every channel; an explicit channel
entry overrides it. What a given channel prints at each level is deliberately not
catalogued here (it grows over time) — raise the channel to `info` or `debug` to
see what it offers.

Every line is formatted as `[neml2:<channel>][<level>] …`, with the channel and
level columns space-padded so they align. Each Newton solve is bracketed by
begin/end separators, so a host that embeds NEML2 solves inside its own
residual/Jacobian evaluations can tell NEML2's output apart from its own. Nested
lines are indented — line-search sub-iterations under their Newton step, inner
linear-solve residuals under theirs, sub-spans by bisection depth:

```
[neml2:newton ][info] ---- begin newton solve ----
[neml2:newton ][dbg ] ITERATION   0, |R| = 3.26e-04, |R0| = 3.26e-04
[neml2:newton ][dbg ]   LS ITERATION   1, min(alpha) = 5.00e-01, ...
[neml2:newton ][dbg ] ITERATION   1, |R| = 1.08e-19, |R0| = 3.26e-04
[neml2:newton ][info] converged (iters=1, reason=rel_tol)
[neml2:newton ][info] ---- end newton solve ----
```

By default (no `NEML2_LOGS` set) every channel sits at `warning`: advisories are
shown, solver/substep traces are not.

## Where the output goes

The default sink splits by level: `info` goes to standard output, `warning` and
`debug` to standard error, one flushed line each. A downstream consumer can
redirect this — see the programmatic API below.

## Programmatic control

A downstream application can set its own **default** levels programmatically; an
`NEML2_LOGS` entry for the same channel still overrides it, so an application
picks its defaults while its end users retain the final say via the environment.

From Python ({mod}`neml2.log`):

```python
import neml2

neml2.log.set_default_level("newton", "info")   # this app's default
neml2.log.set_default_level("all", "silent")     # or quiet everything

# Redirect output:
neml2.log.set_stream(open("solve.log", "w"))      # everything to one stream
neml2.log.set_streams(sys.stdout, sys.stderr)     # split by level
neml2.log.set_sink(lambda level, line: ...)        # full control
```

From C++ (`#include "neml2/csrc/aoti/log.h"`, namespace `neml2::aoti::log`):

```cpp
namespace log = neml2::aoti::log;
log::set_default_level(log::Channel::Newton, log::Level::Info);
log::set_sink([](log::Level lvl, const std::string & line) { /* route to host console */ });
```

A C++ host (for example a finite-element application) typically installs a sink
that forwards NEML2's lines into its own console.

## Command-line tools

`neml2-run` is a thin driver of the library, so its verbosity is controlled
entirely by `NEML2_LOGS` — it has no verbosity flag of its own:

```bash
NEML2_LOGS="newton=info" neml2-run input.i driver
```

`neml2-compile` prints per-file compilation progress as CLI chrome; pass
`-q/--quiet` to suppress it (the final artifact-path summary and any errors are
still printed). This is independent of `NEML2_LOGS`.

The compiler noise from PyTorch itself (Dynamo / Inductor) is out of NEML2's
scope; control it with PyTorch's own `TORCH_LOGS` environment variable.
