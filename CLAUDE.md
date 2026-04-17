# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is NEML2

NEML2 (New Engineering Material model Library, version 2) is a GPU-accelerated, vectorized constitutive modeling library developed at Argonne National Laboratory. It is built on [PyTorch](https://pytorch.org/cppdocs/) as its backend, enabling automatic differentiation, GPU execution, and interoperability with machine learning frameworks.

Key design goals: modularity (models composed from smaller blocks), vectorization (batch evaluation on CPU/GPU), and automatic Jacobian assembly via chain rule or autograd.

## Build System

Requires: CMake 3.26+, C++17, PyTorch/libtorch.

Use `/build` to configure and build. Use `/setup` to install the Python package.

**Available CMake presets:** `dev` (Debug, default), `release`, `cc` (export compile commands), `tsan`, `asan`, `coverage`, `profiling`. The `dev` preset builds to `build/dev/`.

Key CMake options (all ON by default in `dev`):
- `NEML2_TESTS` — build test executables
- `NEML2_TOOLS` — build the runner/tools
- `NEML2_WORK_DISPATCHER` — enable MPI work dispatcher
- `NEML2_JSON` / `NEML2_CSV` — enable JSON/CSV input support
- `NEML2_PCH` — precompiled headers (disable with `cc` preset for IDE indexing)

## Running Tests

Use `/test` to run C++ and Python tests. See `CLAUDE_WORKFLOW.md` for filter syntax.

Test binaries (after `/build`): `build/dev/tests/{unit,regression,verification,dispatchers}/`.
Unit test tags correspond to folder path relative to `tests/unit/`.

## Linting

- **C++:** clang-format version 20 — applied to `src/`, `include/`, `tests/`
- **Python:** black with `--line-length 100` — applied to `python/neml2` and `python/tests`

CI enforces both. Post-edit hooks run automatically on every file Claude Code edits:
- formatting checks via `clang-format` / `black`
- targeted unit-test reruns for edited C++ source/header files when a matching test exists

## Developer tools policy

When a required dev tool (e.g. `clang-format`, `clang-tidy`, `black`, `pytest`) is missing:

1. Detect: `command -v <tool>` or `<tool> --version`
2. If missing, install using the appropriate package manager for the platform:
   - macOS: `brew install <tool>`
   - Debian/Ubuntu: `sudo apt-get install -y <tool>`
   - RHEL/Fedora: `sudo dnf install -y <tool>`
   - Conda environment: `conda install -y <tool>` (if `$CONDA_PREFIX` is set)
3. Verify after install, then retry the original command

Only auto-install common, low-risk formatting/testing tools. Do **not** auto-install build toolchain components (`cmake`, `ninja`, MPI libraries) or anything that modifies compiler configuration — ask first for those.

## Architecture

### Core layers

```
include/neml2/   (headers)        src/neml2/   (implementations)
├── base/        Foundation: Factory, Options, Parser, InputFile, LabeledAxis
├── tensors/     Custom tensor types (Scalar, Vec, R2, SR2, ...) wrapping torch::Tensor
├── user_tensors/  User-facing tensor creation
├── models/      Model base class + all constitutive model implementations
├── solvers/     Newton-Raphson and nonlinear solvers
├── equation_systems/  Assembly and solution of implicit systems
├── drivers/     High-level evaluation drivers (TransientDriver, etc.)
├── dispatchers/ Work distribution (single-device and MPI-based)
└── misc/        Utilities
```

### Factory / Registry pattern

All NEML2 objects (models, solvers, drivers) are registered with a global `Factory` via the `register_NEML2_object` macro. Objects are created by name from HIT input files. This is the primary extension point — new models register themselves and become available in input files automatically.

### HIT input files

Model definitions live in HIT format (hierarchical key=value blocks). The `InputFile` / `Parser` classes parse these files into `Options` objects, which the `Factory` uses to instantiate registered objects. This is the main user interface for composing models.

### Model composition and dependency resolution

Individual models declare their input and output variables on a `LabeledAxis`. When models are composed (via `ComposedModel`), `DependencyResolver` automatically determines evaluation order based on which model's outputs feed another's inputs. Models only need to implement `set_value()` (forward pass) and optionally `set_dvalue()` (derivatives); the full Jacobian is assembled automatically via chain rule.

### Tensor system

All tensors carry a *batch shape* (problem parallelism) and a *base shape* (physical dimensionality). Operations automatically broadcast over the batch dimensions. The `LabeledTensor` family maps named axes onto tensor slices, enabling models to access their inputs/outputs by name.

### Python bindings

C++ functionality is exposed via pybind11 extensions in `python/neml2/`. The Python package is fully interoperable with PyTorch — NEML2 tensors are torch tensors with structured semantics.

## Test Organization

Unit tests in `tests/unit/` generally mirror the `include/neml2/` folder structure, but model
tests often use declarative `.i` files under `tests/unit/models/` instead of a dedicated
`test_Foo.cxx`.

When a dedicated Catch2 source file is used, the usual convention is:
- One `TEST_CASE` named after the header/class
- Tagged `[folder/path]` matching its location under `tests/unit/`
- Top-level `SECTION`s per function/method or behavior

Regression tests cover model combinations for result consistency across releases. Verification tests check mathematical correctness.
