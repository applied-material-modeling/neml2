# AGENTS.md

This file is the **Constitution** and single source of truth for repository-specific agent behavior.

All workflows, roles, and procedures are delegated to the `ai/` directory. Files in `GEMINI.md`, `CLAUDE.md`, and `CLAUDE_WORKFLOW.md` are wrappers that must defer to this document and the linked SOPs.

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

## Global Execution Rules

These rules apply to every task. **Zero exceptions.**

1. **Read Before Edit:** Always read relevant files and SOPs before modifying code.
2. **Follow Patterns:** Prefer existing repository patterns over generic AI preferences.
3. **Targeted Changes:** Prefer minimal, surgical edits over broad refactors.
4. **No Hidden Hooks:** Do not rely on automatic hooks. Simulate them explicitly via [Hooks](#hooks).
5. **Stop on Ambiguity:** When a step says "stop" or logic is unclear, report instead of guessing.
6. **SOP Authority:** The linked files in `ai/` are the absolute authorities for their respective tasks.
7. **No Procedure Duplication:** Task execution details must live in `ai/`. `AGENTS.md` may define global rules, priorities, and routing, but should only summarize task procedures rather than duplicate them.

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

- [BUILD-ENGINEER](ai/roles/build-engineer.md): Build failure diagnosis.
- [CODE-WRITER](ai/roles/code-writer.md): Production C++ implementation.
- [DOC-WRITER](ai/roles/doc-writer.md): Documentation authoring.
- [TEST-WRITER](ai/roles/test-writer.md): Test suite authoring.
