# AGENTS.md

This file is the single source of truth for repository-specific agent behavior.

All workflow logic, role boundaries, and post-edit checks live here. Files in `ai/workflows/`,
`ai/skills/`, `ai/hooks/`, `CLAUDE.md`, `GEMINI.md`, and `CLAUDE_WORKFLOW.md` are wrappers or
indexes that must defer to this document rather than define competing behavior.

## Repository Context

NEML2 is a GPU-accelerated, vectorized constitutive modeling library built on PyTorch/libtorch.
Primary design goals:

- Modularity: models are composed from smaller blocks
- Vectorization: batch evaluation on CPU/GPU
- Automatic differentiation and Jacobian assembly

Core source layout:

- `include/neml2/`: headers
- `src/neml2/`: implementations
- `tests/unit/`: unit tests
- `tests/regression/`: regression tests
- `tests/verification/`: verification tests
- `python/neml2/`: Python bindings
- `doc/content/`: narrative docs

## Global Execution Rules

These rules apply to every LLM and every task in this repository.

1. Read the relevant files before editing.
2. Follow the nearest existing repository pattern over generic preferences.
3. Prefer minimal, targeted changes over refactors.
4. Do not rely on automatic hooks, slash commands, or agent orchestration.
5. Simulate every former hook explicitly by following the post-edit procedures in this file.
6. When a step says "stop", stop and report instead of guessing.
7. `AGENTS.md` overrides all other repo-local workflow documents.

## Build Environment

Requirements:

- CMake 3.26+
- C++17
- PyTorch/libtorch

Available CMake presets:

- `dev`: debug build, default
- `release`: optimized build
- `cc`: compile commands export, PCH off
- `tsan`: thread sanitizer
- `asan`: address sanitizer
- `coverage`: coverage instrumentation
- `profiling`: profiling build

Default build output for `dev`: `build/dev/`

## Developer Tool Policy

When a required low-risk developer tool is missing, detect it first with `command -v <tool>` or
`<tool> --version`.

If missing, install using the platform-appropriate package manager:

- macOS: `brew install <tool>`
- Debian/Ubuntu: `sudo apt-get install -y <tool>`
- RHEL/Fedora: `sudo dnf install -y <tool>`
- Conda env: `conda install -y <tool>` if `$CONDA_PREFIX` is set

Auto-install only common formatting and test tools such as `clang-format`, `clang-tidy`, `black`,
or `pytest`.

Do not auto-install build toolchain components such as `cmake`, `ninja`, MPI libraries, or anything
that changes compiler configuration. Ask first.

## Explicit Post-Edit Procedures

Former hooks are now manual, mandatory procedures.

### After Any Code Edit

For edited C++ files (`.h`, `.cxx`, `.cpp`):

1. Locate a usable formatter in this order:
   - `clang-format`
   - `clang-format-20`
   - `clang-format-19`
   - `$(brew --prefix llvm)/bin/clang-format`
   - `/usr/local/opt/llvm/bin/clang-format`
2. If no formatter exists, install a low-risk formatter per the tool policy.
3. Run a formatting check:
   - `clang-format --dry-run -Werror <file>`
4. If the check fails, format the file and re-check.

For edited Python files (`.py`):

1. Ensure `black` is available, installing it if needed under the tool policy.
2. Run:
   - `black --check --line-length 100 <file>`
3. If the check fails, format the file and re-check.

### After Any C++ Source or Header Edit

If the edited file has a matching dedicated Catch2 test file, explicitly run the associated test.

Procedure:

1. If editing `test_Foo.cxx`, treat `Foo` as the source base.
2. Otherwise search `tests/unit/` for `test_<basename>.cxx`.
3. If no matching test file exists, stop this procedure silently.
4. If `build/dev/tests/unit/unit_tests` does not exist, note that tests cannot be rerun until a build exists.
5. Parse the first `TEST_CASE("...")` name from the test file.
6. Run:
   - `build/dev/tests/unit/unit_tests "<TestCaseName>"`

Failure policy:

1. If the failure happens after editing source/header, first assume the test expectations may need updating.
2. Attempt test-file-only fixes for up to 3 rounds total.
3. Re-run the same test after each test-file fix.
4. If the same failure repeats unchanged, stop immediately.
5. If 3 test-side fix attempts fail, only then fix the source and explicitly report that source changes were required.

## Workflow Procedures

These are universal procedures. Any LLM may execute them directly without slash commands.

### BUILD

Purpose: configure and build the C++ backend using a CMake preset.

Inputs:

- Optional preset, default `dev`
- Optional build target, default all
- Optional `--reconfigure`

Valid presets:

- `dev`
- `release`
- `cc`
- `tsan`
- `asan`
- `coverage`
- `profiling`

Procedure:

1. If `build/<preset>/` does not exist, or `--reconfigure` is requested, run:
   - `cmake --preset <preset> -GNinja -S .`
2. Build:
   - `cmake --build --preset <preset> [--target <target>]`
3. If a header path from an error message is unclear, search for it under:
   - `include/neml2/**/<Filename>.h`
4. On success, report the output path.

Light repair loop, maximum 3 attempts total including the first build:

1. Extract the first meaningful compiler error.
2. If the error is trivial, limited to 1-2 files, apply the minimal fix and retry.
3. Trivial means cases such as:
   - missing `#include`
   - obvious typo
   - missing `register_NEML2_object(...)`
4. If the error is non-trivial, do not modify code. Report the file, line, and required fix.
5. If the exact same error repeats, stop immediately.
6. After 3 total attempts, stop and report the outstanding error.
7. If unresolved, recommend the `FIX-BUILD` procedure.

Named targets commonly used with `dev`:

- `unit_tests`
- `regression_tests`
- `verification_tests`
- `dispatcher_tests`

### FIX-BUILD

Purpose: diagnose and repair an already failing build.

Procedure:

1. Run:
   - `cmake --build --preset dev`
2. If the build succeeds, report success and stop.
3. If it fails, extract the first meaningful compiler or linker error.
4. Apply the `BUILD-ENGINEER` role procedure below.
5. Rebuild after each targeted fix, with a hard limit of 3 build attempts total.
6. If resolved, summarize what changed and confirm the build is clean.
7. If unresolved, report the exact outstanding error and why auto-repair stopped.

### TEST

Purpose: run C++ and/or Python tests.

Input dispatch:

- no argument: all C++ unit tests and all Python tests
- `cpp` or `c++`: all four C++ suites
- `python` or `py`: Python tests only
- `unit`: C++ unit tests only
- `regression`: regression suite only
- `verification`: verification suite only
- `dispatcher`: dispatcher suite only
- `[tag]`: Catch2 tag filter
- any other string: Catch2 test name or pytest node id

Preset resolution:

1. If the first argument is a known preset, use it and remove it from the remaining test argument.
2. Otherwise default to `dev`.
3. If `build/<preset>/tests/unit/unit_tests` is missing:
   - search other `build/*/tests/unit/unit_tests`
   - if one is found, use it
   - if several are found, use the most recently modified one
   - if none are found, stop and report that a build is required first

C++ commands:

- unit: `./build/<preset>/tests/unit/unit_tests`
- tag: `./build/<preset>/tests/unit/unit_tests "[tag]"`
- named test: `./build/<preset>/tests/unit/unit_tests "TestName"`
- regression: `./build/<preset>/tests/regression/regression_tests`
- verification: `./build/<preset>/tests/verification/verification_tests`
- dispatcher: `./build/<preset>/tests/dispatchers/dispatcher_tests`

Python environment resolution:

1. Find a Python interpreter that imports both `torch` and `neml2`.
2. If none exists, find one that imports `torch`.
3. If `torch` exists but `neml2` does not, run `SETUP` once, then retry interpreter discovery.
4. If an interpreter exists but lacks `pytest`, install `pytest`.
5. If no interpreter with `torch` exists, stop and report that `torch` must be installed manually.
6. Do not auto-install `torch`.

Python commands:

- full suite: `<py> -m pytest -v python/tests`
- single file or node: `<py> -m pytest -v python/tests/test_foo.py`
- single function: `<py> -m pytest -v python/tests/test_foo.py::test_bar`

Reporting:

Always report totals, passed, failed, elapsed time, and failing test names with error messages.

Failure triage:

1. If the problem is clearly test logic, fix the test.
2. If the problem is clearly production code, report the file and lines; do not modify production code during test triage.
3. If unclear, stop and report the ambiguity instead of guessing.

When production-code consistency matters, inspect all relevant locations:

- `include/neml2/...`
- `src/neml2/...`
- `tests/unit/...`
- relevant `CMakeLists.txt`

### DOCS

Purpose: add or improve documentation for a file, class, or module.

Priority order:

1. `expected_options()` in `.cxx`
2. per-option `.doc()` strings
3. brief header Doxygen matched to local style
4. `doc/content/` narrative pages

Procedure:

1. Read the target `.cxx` and locate `expected_options()`.
2. Improve `options.doc()` so it states what the object computes, from which inputs, and its governing equation.
3. Improve per-option `.doc()` strings with units, meaning, and constraints.
4. Use `\f$ ... \f$` for LaTeX in `.cxx` docstrings.
5. Read at least 3 neighboring headers before touching a header comment.
6. If neighbors use class comments, match their local form, usually a brief `@brief`.
7. Do not add `### Variables`, `@param`, or HIT examples to headers.
8. For new models or constitutive laws, update the appropriate `doc/content/modules/*.md` page.
9. Include governing equations, variable definitions, parameter descriptions, units, and a minimal HIT example via `@list-input`.
10. After editing documentation, run `DOCS-VERIFY`.

### DOCS-VERIFY

Purpose: verify documentation builds cleanly.

Procedure:

1. Run:
   - `./doc/scripts/examples.py --log-level INFO --cmake-configure-args="-GNinja"`
   - `./doc/scripts/genhtml.py --log-level INFO`
2. Inspect:
   - `build/doc/doxygen.html.log`
   - `build/doc/doxygen.python.log`
   - `build/doc/syntax.err`
3. Report:
   - full `syntax.err` contents if non-empty
   - any warnings or errors from the doxygen logs
   - otherwise state that documentation builds cleanly
4. Do not auto-fix doc build failures in this procedure.

### IMPLEMENT

Purpose: implement a new NEML2 model end-to-end without multi-agent orchestration.

Execute these steps in order. Stop on failure.

Step 0. Find a design spec.

1. Search `design/` for matching `.md` or `.pdf` specs.
2. If one spec matches, use it as the source of truth.
3. If multiple specs in the same directory match, read all of them and implement each variant.
4. If variants share a common structure, introduce a shared base class only if justified by the specs.
5. If matching specs span different directories, choose the closest module and model match.
6. If no spec exists, proceed using the user request and similar existing models.

Step 1. Write production code using the `CODE-WRITER` role procedure.

Step 2. Run `BUILD` with preset `dev`.

Step 3. Write tests using the `TEST-WRITER` role procedure.

Step 4. Build test targets and run focused tests:

1. `cmake --build --preset dev --target unit_tests`
2. run the most relevant test or tests

Step 5. If tests fail:

1. fix test logic only when clearly appropriate
2. if failure is in production code, stop and report rather than silently changing production logic here
3. stop if the same failure repeats unchanged
4. stop after 3 rounds of test-side fixes

Step 6. Complete documentation using the `DOC-WRITER` role procedure.

Step 7. Remind the user to run or review `DOCS-VERIFY`.

### SETUP

Purpose: set up the Python development environment.

Procedure:

1. Use the requested Python executable or default to `python3`.
2. Install:
   - `pip install ".[dev]" -v`
3. Verify:
   - `python3 -c "import neml2; print('neml2', neml2.__version__, 'installed OK')"`
4. Confirm pytest discovery:
   - `pytest --collect-only python/tests -q`
5. If extension compilation fails, recommend doing a C++ build first with the `cc` preset.

### NAMING

Purpose: audit naming consistency without auto-fixing.

If no file list is provided, audit files changed relative to `main`.

Ignore `contrib/`.

Checks:

1. File name vs primary class name in `.h`
2. `register_NEML2_object(...)` consistency in `.cxx`
3. HIT `type =` names in `.i` files vs registered class names
4. private/protected member variable leading underscore convention

Output format:

`FILE:LINE  [rule]  found: actual  expected: expected`

Report only concrete mismatches. Skip uncertain cases silently. Print a total violation count.

### SMOKE-TEST

Purpose: verify the workflow still works without `.claude/`.

Procedure:

1. Record changed files before and after any step that edits files.
2. Revert temporary smoke-test edits after each such step.
3. Check:
   - `BUILD` variants
   - `TEST` variants
   - explicit post-edit formatting procedure
   - `TEST-WRITER` procedure on an existing model
   - `BUILD-ENGINEER` procedure on new-file registration reasoning
   - `DOC-WRITER` procedure on an existing model
   - full `IMPLEMENT` pipeline on a disposable stub model
4. Report `PASS`, `FAIL`, or `PARTIAL` for each check and summarize at the end.

## Skill Procedures

These replace former specialized agents. Execute them as normal sequential guidance.

### BUILD-ENGINEER

Purpose: diagnose and fix NEML2 build failures with minimal changes.

Procedure:

1. Read the full compiler or linker error first.
2. Read the relevant header and source before editing.
3. Before touching any `CMakeLists.txt`, check whether sources are collected by `GLOB` or `GLOB_RECURSE`.
4. If globbing is used, do not edit `CMakeLists.txt`; ensure the file is placed correctly and that the source ends with `register_NEML2_object(ClassName)`.
5. If sources are explicit, add the file to `target_sources(...)`.
6. Apply the minimal fix.
7. Rebuild.
8. If a new error appears, apply one more targeted fix and rebuild.
9. Stop if the same error repeats or 3 build attempts are exhausted.
10. Never modify `CMakePresets.json` without explicit approval.

Common patterns to check:

- missing source registration
- missing `register_NEML2_object`
- PCH/include-order issues
- libtorch ABI mismatches
- missing MPI when dispatcher support is enabled

### CODE-WRITER

Purpose: implement or modify production C++ code only.

Responsibilities:

- headers under `include/neml2/`
- sources under `src/neml2/`
- minimal `expected_options()` docstrings
- build-system wiring only if explicitly required by non-glob CMake

Rules:

1. Do not write tests here.
2. Do not write full documentation here.
3. Add `register_NEML2_object(ClassName)` at the bottom of every new `.cxx`.
4. Write only one-line `.doc()` strings in `expected_options()` as a minimum.
5. When assigning from `Variable<T>`, use `operator()` to get the tensor value rather than copy-assigning the variable object.
6. Check whether derivative-disabled `.i` tests should be re-enabled after derivative fixes.

Workflow:

1. Read similar nearby models first.
2. Implement the minimal correct code.
3. Check CMake wiring only after inspecting how sources are collected.
4. Hand off to `BUILD`, then `TEST-WRITER`, then `DOC-WRITER`.

### DOC-WRITER

Purpose: write documentation only.

Rules:

1. Primary target is `expected_options()` in `.cxx`.
2. Header comments are brief and only added if local style supports them.
3. Read at least 3 neighboring headers before modifying a header.
4. For new models, update `doc/content/modules/` or create a suitable new page.
5. Do not move detailed formulation text into headers.
6. For Python docstrings, use Google style with `Args:`, `Returns:`, and `Raises:` when applicable.

Workflow:

1. Read the `.cxx`.
2. Fill in missing or weak `.doc()` strings.
3. Infer header style from neighbors and add only minimal header docs if warranted.
4. Update narrative docs for new models.
5. Run `DOCS-VERIFY`.

### TEST-WRITER

Purpose: write tests only.

Decision order:

1. Read the model header.
2. Read the surrounding test directory.
3. Match the nearest existing local pattern.

Rules for `Model` subclasses:

1. Default to declarative `.i` tests under `tests/unit/models/`.
2. Use `ModelUnitTest` unless procedural logic cannot be expressed declaratively.
3. Write a dedicated `test_*.cxx` only for procedural behaviors such as:
   - exception handling
   - parser/factory wiring checks
   - axis metadata checks
   - multi-step or driver workflows
4. Do not edit `CMakeLists.txt` if globbing already discovers new `.cxx` files.

Derivative-check rules:

1. `ModelUnitTest` already checks values, first derivatives, and second derivatives.
2. Do not write a hand-coded derivative section when `ModelUnitTest` is sufficient.
3. Any `check_values = false`, `check_derivatives = false`, or `check_second_derivatives = false` must have an explanatory comment immediately above it.
4. If a point is degenerate for forward finite differences, keep the check disabled there, explain why, and add a non-degenerate coverage case elsewhere.

Workflow:

1. Read the target header.
2. Read a neighboring test in the same area.
3. Choose `.i` or `test_*.cxx` based on the behavior.
4. Write a complete test with no TODOs.
5. Build and run the targeted test.
