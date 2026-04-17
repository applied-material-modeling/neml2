Smoke-test the NEML2 Claude Code workflow end-to-end.

Run all checks in order. For any step that creates or modifies files, record changed files
with `git diff --name-only` before and after, then run `git restore <files>` to revert.
Report PASS / FAIL / PARTIAL for each step. Print a summary table at the end.

---

## 1. Build command

Run these three variants and verify each succeeds:

- `/build` — configure (if needed) + build all targets
- `/build dev unit_tests` — build only the unit test binary
- `/build --reconfigure` — force reconfigure then rebuild

PASS if all three complete without compiler errors.

---

## 2. Test command

Run these three variants and verify each reports no failures:

- `/test unit` — all C++ unit tests
- `/test [tensors]` — unit tests filtered by `[tensors]` tag
- `/test python` — all Python tests via pytest

PASS if all assertions pass (skips due to missing CUDA are acceptable).

---

## 3. Hook (post-edit linting)

Make a trivial, style-clean edit to any `.h` file. Observe the hook output.
Then `git restore` the file.

- PASS: `[neml2] clang-format OK: include/neml2/...` appears
- PARTIAL: hook fires but exits silently — `clang-format` not installed; note which fallback binary is missing
- FAIL: hook does not fire at all (check `.claude/settings.json`)

---

## 4. test-writer agent

Record changed files before: `git diff --name-only`

Invoke the test-writer agent for:
`include/neml2/models/solid_mechanics/elasticity/LinearIsotropicElasticity.h`

PASS if the agent:
- Reads the surrounding test directory before deciding what to write
- Produces a `.i` file under `tests/unit/models/solid_mechanics/elasticity/` using `ModelUnitTest` as the driver (`.cxx` is acceptable only if it gives a clear procedural reason)
- Includes analytic reference values in `output_*_names/values`
- Does NOT hand-code a finite-difference derivative section (ModelUnitTest handles that)
- Does NOT write a `test_LinearIsotropicElasticity.cxx` without justification
- No TODO comments

After evaluating: `git restore` all files changed by this step.

---

## 5. build-engineer agent

Invoke the build-engineer agent to register `src/neml2/models/Foo.cxx` in the build system.

PASS if the agent:
- Detects that `GLOB_RECURSE` is used
- Reports that no `CMakeLists.txt` edit is needed
- Describes correct file placement instead

FAIL if the agent makes any `CMakeLists.txt` edit.

No file revert needed (agent should not modify any files on PASS).

---

## 6. doc-writer agent

Record changed files before: `git diff --name-only`

Invoke the doc-writer agent for:
`include/neml2/models/solid_mechanics/elasticity/LinearIsotropicElasticity.h`

PASS if the agent:
- Reads the `.cxx` and fills in `options.doc()` and per-option `.doc()` strings in `expected_options()` (primary target)
- Reads ≥3 neighboring headers and infers local style before touching the header
- Header comment is brief (`/** @brief ... */`, 2–4 lines max) or absent if neighbors have none
- Does NOT add `### Physics`, `### Variables`, `### HIT input example`, or `@param` blocks

After evaluating: `git restore` all files changed by this step.

---

## 7. code-writer agent + /implement pipeline

Invoke the **code-writer agent** to implement a minimal model stub:
`"A no-op model SmokeTestModel that takes a scalar input and returns it unchanged"`

PASS if the agent:
- Produces `include/neml2/models/SmokeTestModel.h` and `src/neml2/models/SmokeTestModel.cxx`
- Ends the `.cxx` with `register_NEML2_object(SmokeTestModel)`
- Does NOT edit `CMakeLists.txt` (GLOB_RECURSE handles it)
- Writes minimal one-line `.doc()` strings only

After evaluating, revert with:
```bash
git restore include/ src/
git clean -fd include/ src/
```

The full `/implement` pipeline (code-writer → `/build dev` → doc-writer → test-writer → tests)
is intentionally kept separate — it creates files across `include/`, `src/`, and `tests/`.
Run it manually when needed and revert with:
```bash
git restore include/ src/ tests/
git clean -fd include/ src/ tests/
```

---

## Summary table

| # | Check | Result | Notes |
|---|-------|--------|-------|
| 1 | /build variants | | |
| 2 | /test variants | | |
| 3 | clang-format hook | | |
| 4 | test-writer agent | | |
| 5 | build-engineer agent | | |
| 6 | doc-writer agent | | |
| 7 | code-writer agent | | |

Flag any PARTIAL or FAIL with a one-line diagnosis and suggested fix.
