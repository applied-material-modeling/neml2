# AGENTS.md

This file is the **Constitution** and single source of truth for repository-specific agent behavior.

All workflows, roles, and procedures are delegated to the `ai/` directory. Files in `GEMINI.md` and `CLAUDE.md` are wrappers that must defer to this document and the linked SOPs.

## Repository Context

NEML2 is a GPU-accelerated, vectorized constitutive modeling library built on PyTorch/libtorch.
Primary design goals:
- Modularity: models are composed from smaller blocks
- Vectorization: batch evaluation on CPU/GPU
- Automatic differentiation and Jacobian assembly

### Source Layout
- `include/neml2/`: headers
- `src/neml2/`: implementations
- `tests/unit/`: unit tests
- `tests/regression/`: regression tests
- `tests/verification/`: verification tests
- `python/neml2/`: Python bindings
- `doc/content/`: narrative docs

### Core API Map

Read these files **before** implementing or modifying any model. They define the interfaces every model depends on.

| Concern | Header |
|---|---|
| Base model class (`set_value`, variable declaration, parameter declaration) | `include/neml2/models/Model.h` |
| Variable template (`Variable<T>`, `is_dependent()`, `.d()`, history/rate/residual name helpers) | `include/neml2/models/Variable.h`, `include/neml2/models/VariableBase.h` |
| Option registration (`expected_options`, `.doc()`, `add_input/add_output/add_parameter/add_buffer`) | `include/neml2/base/OptionSet.h` |
| Scalar tensor | `include/neml2/tensors/Scalar.h` |
| 3-vector | `include/neml2/tensors/Vec.h` |
| Symmetric 2nd-order tensor | `include/neml2/tensors/SR2.h` |
| Full 2nd-order tensor | `include/neml2/tensors/R2.h` |
| All math functions (where, inv, dev, norm, …) | `include/neml2/tensors/functions/` |

### Variable Naming

Use bare variable names everywhere — both in C++ (`declare_input_variable<T>("strain_rate")`) and in `.i` files (`input_Scalar_names = 'strain_rate'`). No subspace prefixes.

- **History (old-state)** variables use a `~N` suffix where `N` is the lag order. Use the `history_name(name, /*nstep=*/N)` helper from `Variable.h` rather than building the string yourself.
- **Rate** variables use the `rate_name(name)` helper.
- **Residual** of unknown `u` defaults to `u_residual` via `residual_name(name)`.
- `NonlinearSystem` requires explicit `unknowns = '...'` in input files; residuals default to `<unknown>_residual` per unknown.

### Model Boilerplate Templates

`ai/templates/` contains copy-paste starting points for new models:

- `SkeletonModel.h` — header with correct license, pragma once, variable/parameter members
- `SkeletonModel.cxx` — implementation with `register_NEML2_object`, `expected_options`, constructor, and `set_value` structure
- `SkeletonModelTest.i` — `ModelUnitTest` fixture with `[Tensors]` block, derivative check flags, and usage comments

Before writing any new model file, read the nearest existing model in the same subdirectory to confirm local conventions, then use the template as a structural guide — not a copy.

## Global Execution Rules

These rules apply to every task. **Zero exceptions.**

1. **Read Before Edit:** Always read relevant files and SOPs before modifying code.
2. **Follow Patterns:** Prefer existing repository patterns over generic AI preferences.
3. **Targeted Changes:** Prefer minimal, surgical edits over broad refactors.
4. **No Hidden Hooks:** Do not rely on automatic hooks. Simulate them explicitly via [Hooks](#hooks).
5. **Stop on Ambiguity:** When a step says "stop" or logic is unclear, report instead of guessing.
6. **SOP Authority:** The linked files in `ai/` are the absolute authorities for their respective tasks.
7. **No Procedure Duplication:** Task execution details must live in `ai/`. `AGENTS.md` may define global rules, priorities, and routing, but should only summarize task procedures rather than duplicate them.
8. **Complete Workflows:** When a `/workflow` is invoked, execute ALL numbered steps in sequence. Do not stop at an intermediate step because it "feels done" (e.g., tests passing is not the end of IMPLEMENT — Steps 5 and 6 remain). A workflow is complete only when its final step is done.
9. **Context Resumption:** If this session is a continuation of a prior session (summarized context), re-read the active workflow at the start, identify the last completed step from the summary, and continue from the next uncompleted step. Do not re-derive what was done — trust the summary and move forward.

## Build Environment

Requirements: CMake 3.26+, C++17, PyTorch/libtorch.

### CMake Presets
- `dev`: debug build (default). Output: `build/dev/`
- `release`: optimized build
- `cc`: compile commands export, PCH off
- `tsan`/`asan`: sanitizers
- `coverage`: instrumentation
- `profiling`: profiling build

## Developer Tool Policy

1. **Detection:** Check for tools with `command -v <tool>` or `<tool> --version`.
2. **Installation:** Use `brew` (macOS), `apt-get` (Debian), `dnf` (RHEL), or `conda`.
3. **Scope:** Auto-install only low-risk tools (`clang-format`, `black`, `pytest`).
4. **Restriction:** Do not auto-install toolchain components (`cmake`, `ninja`, MPI). Ask first.

---

## Hooks (Manual Post-Edit Procedures)

AI agents must execute these explicitly after modifications.

- [Post-Edit](ai/hooks/post-edit.md): Formatting (clang-format, black).
- [Test-on-Edit](ai/hooks/test-on-edit.md): Targeted test execution for C++ edits.

## Workflow Procedures (SOPs)

- **Default implementation routing:** "Implement/add/create a model" means run
  [IMPLEMENT](ai/workflows/implement.md), unless told otherwise.
- [BUILD](ai/workflows/build.md): Configure and build the C++ backend.
- [FIX-BUILD](ai/workflows/fix-build.md): Diagnose and repair a failing build.
- [TEST](ai/workflows/test.md): Run C++ and/or Python tests.
- [DOCS](ai/workflows/docs.md): Add or improve documentation.
- [DOCS-VERIFY](ai/workflows/docs-verify.md): Verify documentation builds.
- [IMPLEMENT](ai/workflows/implement.md): End-to-end new model implementation.
- [SETUP](ai/workflows/setup.md): Python environment setup.
- [NAMING](ai/workflows/naming.md): Audit naming consistency.
- [SMOKE-TEST](ai/workflows/smoke-test.md): Workflow integrity verification.

## Role Guides

Specialized guidance for specific agent roles.

- [NEML2-GUIDELINES](ai/roles/neml2-guidelines.md): Repository-specific C++ idioms and gotchas to read before editing NEML2 code.
- [BUILD-ENGINEER](ai/roles/build-engineer.md): Build failure diagnosis.
- [CODE-WRITER](ai/roles/code-writer.md): Production C++ implementation.
- [DOC-WRITER](ai/roles/doc-writer.md): Documentation authoring.
- [TEST-WRITER](ai/roles/test-writer.md): Test suite authoring.

## Templates

Boilerplate starters for new models — always read the nearest existing model first to confirm local style.

- [SkeletonModel.h](ai/templates/SkeletonModel.h): Header with license, `expected_options`, variable/parameter members.
- [SkeletonModel.cxx](ai/templates/SkeletonModel.cxx): Implementation with `register_NEML2_object`, constructor, `set_value`.
- [SkeletonModelTest.i](ai/templates/SkeletonModelTest.i): `ModelUnitTest` fixture with `[Tensors]` block and derivative flags.
