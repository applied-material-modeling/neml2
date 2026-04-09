# Claude Code Workflow for NEML2

This document explains how to use Claude Code commands and agents in this repository.

---

## Overview

The workflow is organized into three layers:

| Layer | What it is | Examples |
|-------|-----------|---------|
| **Commands** (`/name`) | Slash commands Claude Code executes | `/build`, `/test`, `/implement` |
| **Agents** | Specialized sub-agents invoked by Claude Code | `code-writer`, `doc-writer`, `test-writer`, `build-engineer` |
| **Hook** | Automatic post-edit linting | runs `clang-format` / `black` on every Edit |

Claude Code orchestrates everything. Agents are invoked when a task matches their specialty.

---

## Commands

### `/build [preset] [target]`
Build the C++ backend using CMake.

```
/build                   # configure + build dev preset
/build dev unit_tests    # build only the unit test binary
/build release           # optimized build
/build --reconfigure     # force reconfigure then build
```

Presets: `dev` (default), `release`, `cc`, `tsan`, `asan`, `coverage`, `profiling`.

On failure, attempts up to 3 auto-repairs for trivial errors (missing `#include`, missing
`register_NEML2_object`). Stops and reports for non-trivial errors.

---

### `/fix-build`
Repair mode for a failed build. Invokes the build-engineer agent to diagnose and fix the
current build failure. Use this when `/build` has exhausted its own repair loop.

```
/fix-build
```

| Command | Use when |
|---------|----------|
| `/build` | Normal build — configure, build, light inline repair |
| `/fix-build` | Build already failing — delegate to build-engineer for deeper repair |

---

### `/test [filter]`
Run C++ and/or Python tests.

```
/test                    # all unit tests + all Python tests
/test unit               # C++ unit tests only
/test regression         # C++ regression tests only
/test [tensors]          # unit tests filtered by Catch2 tag
/test LabeledAxis        # single test case by name
/test python             # Python tests via pytest
```

Reports: total / passed / failed / elapsed time. For each failure, triages whether
the root cause is test logic or production code before acting.

---

### `/docs [path or class name]`
Add or improve documentation.

Works in priority order:
1. `expected_options()` `.doc()` strings in `.cxx` (feeds the website)
2. Per-option `.doc()` strings
3. Header `/** @brief */` comment matched to local style
4. `doc/content/` narrative pages

---

### `/docs-verify`
Build the documentation and check for errors.

Runs the doc generation scripts and checks `doxygen.html.log`, `doxygen.python.log`,
and `syntax.err`. Reports issues without auto-fixing them.

---

### `/implement [description]`
Implement a new model class end-to-end (the full pipeline):

```
/implement traction-separation law
/implement CubicIsotropicHardening
```

Runs in order — stops if any step fails:

1. **code-writer agent** — header + `.cxx` + `register_NEML2_object`; no CMakeLists.txt edit needed
2. **`/build dev`** — verify it compiles
3. **doc-writer agent** — complete `expected_options()` docstrings
4. **test-writer agent** — write unit tests; run `/build dev unit_tests` + `/test "ClassName"`
5. **Remind** — suggests `/docs-verify`

---

### `/setup [python path]`
Set up the Python development environment.

```
/setup                          # pip install ".[dev]" with default python3
/setup /path/to/venv/bin/python # custom Python executable
```

---

### `/naming [files]`
Audit naming conventions in given files, or all files changed relative to `main` if no argument.

Reports violations (file, line, rule, found, expected) without auto-fixing.

---

### `/smoke-test`
Verify the entire Claude Code workflow is working correctly. Runs all 7 checks and
reverts any files written during the test. See the checklist for expected PASS criteria.

---

## Agents

Agents are invoked by Claude Code for specialized tasks. Each has a defined scope.

### `code-writer`
**Implements production C++ code. Does not write tests or full documentation.**

- Writes `include/neml2/<path>/Foo.h` and `src/neml2/<path>/Foo.cxx`
- Adds `register_NEML2_object(Foo)` at the bottom of every new `.cxx`
- Writes minimal one-line `.doc()` strings in `expected_options()` only — full docs are doc-writer's job
- Checks CMake wiring (GLOB vs explicit) before touching any `CMakeLists.txt`
- Invoked as Step 1 of `/implement`

### `doc-writer`
**Writes documentation only. Does not touch production logic or tests.**

- Primary: `expected_options()` `.doc()` strings in `.cxx` (shown on the website)
- Secondary: brief `/** @brief */` header comment, matched to the style of neighboring files
- Does NOT write `### Physics`, `### Variables`, or HIT example sections in headers
- Always reads ≥3 neighboring headers to infer local style before writing anything

### `test-writer`
**Writes Catch2 unit tests and pytest tests. Does not touch production code.**

- Writes `tests/unit/<path>/test_Foo.cxx` mirroring `include/neml2/<path>/Foo.h`
- Mandatory: `set_value` (analytic reference) and `set_dvalue` (finite diff at `atol=rtol=1e-5`) sections
- No TODO stubs — every section is fully implemented
- Locates headers via Glob if the provided path doesn't exist directly

### `build-engineer`
**Diagnoses and fixes build failures. Makes minimal, targeted changes.**

- Before touching any CMakeLists.txt, checks whether `GLOB_RECURSE` is used
- If glob is used: explains that placing the file in the correct directory is sufficient
- Only edits CMakeLists.txt if sources are listed explicitly
- Stops after 3 build attempts and reports outstanding errors

---

## Agent boundaries

| Task | Who does it |
|------|------------|
| Write production code (`.h`, `.cxx`) | `code-writer` agent |
| Write `expected_options()` `.doc()` strings | `doc-writer` agent |
| Write header Doxygen comments | `doc-writer` agent |
| Write unit / regression tests | `test-writer` agent |
| Diagnose and fix build failures | `build-engineer` agent |
| Build the project | `/build` command |
| Run tests | `/test` command |
| Sequence the full pipeline | `/implement` command |

---

## Post-edit hook

Every time Claude Code edits a `.h`, `.cxx`, or `.py` file, a hook runs automatically:

- **C++ files:** `clang-format --dry-run -Werror` (tries `clang-format`, then `clang-format-20`,
  `clang-format-19`, brew llvm path, `/usr/local/opt/llvm/bin/clang-format`)
  - Clean: `[neml2] clang-format OK: include/neml2/...`
  - Issue: `[neml2] Style issue — run: clang-format -i <file>`
- **Python files:** `black --check --line-length 100`
  - Clean: `[neml2] black OK: python/neml2/...`
  - Issue: `[neml2] Style issue — run: black --line-length 100 <file>`

If no formatter is installed, the hook exits silently.

---

## Developer tools policy

If a required dev tool (`clang-format`, `clang-tidy`, `black`, `pytest`) is missing, Claude Code
will attempt to install it using the appropriate package manager and retry. See `CLAUDE.md`
for the full policy.

---

## Example workflows

### Add a new constitutive model

```
/implement PowerLawCreep
```

This runs the full pipeline: code-writer → build → doc-writer → test-writer → remind about docs-verify.

### Debug a failing test

```
/test [models/solid_mechanics]
```

Claude Code triages each failure (test logic vs. production code) before modifying anything.

### Add documentation to an existing model

```
/docs src/neml2/models/solid_mechanics/VoceIsotropicHardening.cxx
```

### Check naming conventions before a PR

```
/naming
```

Audits all files changed relative to `main`.

### Verify the workflow is healthy after setup

```
/smoke-test
```

---

## File layout reference

```
.claude/
├── settings.json          # hook configuration
├── hooks/
│   └── post_edit.sh       # clang-format / black on every edit
├── commands/
│   ├── build.md           # /build
│   ├── fix-build.md       # /fix-build (repair mode)
│   ├── test.md            # /test
│   ├── docs.md            # /docs
│   ├── docs-verify.md     # /docs-verify
│   ├── implement.md       # /implement (full pipeline)
│   ├── setup.md           # /setup
│   ├── naming.md          # /naming
│   └── smoke-test.md      # /smoke-test
└── agents/
    ├── code-writer.md     # production code agent
    ├── doc-writer.md      # documentation agent
    ├── test-writer.md     # test generation agent
    └── build-engineer.md  # build diagnosis agent
```
