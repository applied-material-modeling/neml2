# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NEML2 is a C++17 material modeling library that vectorizes constitutive model evaluation on CPU/GPU using LibTorch as the tensor backend. Models are composed from small reusable pieces, the framework auto-resolves dependencies between them, and most users interact via HIT input files (the same format used by MOOSE) plus either the C++ API, the Python bindings, or the `neml2-run` CLI.

## Build & develop (C++)

CMake presets in `CMakePresets.json` are the canonical entry point. Use `--preset` rather than passing raw flags so the build/install dirs (`build/<preset>`, `installed/<preset>`) stay consistent.

```bash
cmake --preset dev          # configure (Debug + tests + tools + dispatcher + JSON + CSV)
cmake --build --preset dev  # build the `tests` and `tools` targets
```

Other notable presets: `release`, `cc` (exports `compile_commands.json` and symlinks it into the repo root), `coverage`, `asan`, `tsan`, `profiling`. CI also exercises a `-DNEML2_PCH=OFF` build (no precompiled headers) and a "no optional" build that disables `NEML2_JSON`, `NEML2_CSV`, and `NEML2_WORK_DISPATCHER` — keep those guarded with `#ifdef NEML2_HAS_*` where relevant.

The build pulls dependencies via `cmake/Modules/Findtorch.cmake` (PyTorch must already be importable in Python; libTorch is found in site-packages by default) and downloads/builds `nmhit`, `Catch2`, `nlohmann_json`, `csvparser`, `argparse`, `gperftools` into `contrib/` as needed. The `nmhit` submodule is auto-initialized on first configure.

### Python package

```bash
pip install ".[dev]" -v       # editable + dev extras (pytest, pre-commit, …)
```

This drives a `scikit-build-core` build of the `pyneml2` extension under `build/<wheel_tag>/`. The Python package version, libTorch version, and other pinned deps are tracked in `dependencies.yaml` — use `python scripts/dep_manager.py {check|list|bump DEP.FIELD VALUE}` rather than editing version strings by hand. Files reference their dep with a `# dependencies: NAME.FIELD` annotation immediately above the version literal.

## Tests

Tests are Catch2 binaries built into `build/<preset>/tests/`. Run individual binaries directly to keep iteration fast:

```bash
./build/dev/tests/unit/unit_tests          # most unit tests (base/, models/, solvers/, …)
./build/dev/tests/unit/tensor_tests        # tensor-class unit tests
./build/dev/tests/regression/regression_tests
./build/dev/tests/verification/verification_tests
./build/dev/tests/dispatchers/dispatcher_tests   # only when NEML2_WORK_DISPATCHER=ON
```

Catch2 selectors apply: `./unit_tests "[base/Factory]"` runs by tag, `./unit_tests -# "test_Foo"` runs a single test case. To exercise CUDA paths pass `--devices cuda` (CI sets `TEST_DEVICE_ARGS` accordingly on the `gpu_runner`).

For spot-checks after a code change, run individual scenarios — not the full suite. `regression_tests` and `verification_tests` each have one `TEST_CASE` per submodule and discover scenarios as Catch2 sections, so target a single scenario via `-c` (the `add-regression` and `add-verification` skills document the exact pattern):

```bash
./build/dev/tests/regression/regression_tests "solid mechanics" -c "viscoelasticity/maxwell/model.i"
./build/dev/tests/verification/verification_tests "solid mechanics" -c "viscoelasticity/zener/zener.i"
```

Reserve unfiltered `regression_tests` / `verification_tests` runs for final confirmation.

Python tests:

```bash
pytest -v python/tests
pytest -v python/tests/test_Model.py::test_forward  # single test
```

Test layout under `tests/unit/` mirrors `include/neml2/`. When adding a `Model` subclass, use the `/add-model` skill; for any other header, use `/add-test`. Both encode the `tests/unit/README.md` conventions (one `TEST_CASE` per file, tag = directory).

## Documentation

The doc build pipeline (`neml2-stub` → `doc/scripts/examples.py` → `doc/scripts/genhtml.py`) requires a non-editable `neml2` install. Use the `/build-docs` skill, or see `doc/content/tutorials/contributing.md` for the full workflow.

## CLI tools

When `NEML2_TOOLS=ON` (default in most presets), `src/tools/` builds executables: `neml2-run`, `neml2-inspect`, `neml2-diagnose`, `neml2-syntax`, `neml2-time`. The Python package re-exposes the same entry points as `neml2-run`, `neml2-inspect`, etc., wired through `python/neml2/_cli.py`.

## Architecture

The C++ library is split into co-equal submodules under `src/neml2/`, each producing a separate `libneml2_<name>` shared library and a public header tree under `include/neml2/<name>/`. The umbrella `libneml2` (target `neml2`) just `PUBLIC`-links them all.

- `base/` — input parsing (HIT via `nmhit`), `Factory`, `Registry`, `OptionSet`, `NEML2Object` base, settings, diagnostics, tracing.
- `tensors/` — domain tensor classes (`Scalar`, `Vec`, `R2`, `SR2`, `Rot`, `Quaternion`, fourth-order `R4`/`SSR4`/`WSR4`/…) wrapping `torch::Tensor` with batched semantics + JIT helpers (`tensors/jit.cxx`, `tensors/functions/`).
- `models/` — the composable forward operators. `Model` is the base; `ComposedModel` glues children together via the dependency graph in `models/DependencyResolver.h`; `ImplicitUpdate` wraps a residual model in a Newton solve with optional `Predictor`. Domain libraries live in `models/{solid_mechanics,crystallography,chemical_reactions,phase_field_fracture,porous_flow,finite_volume,common}/`.
- `solvers/` — `NonlinearSolver` (Newton, NewtonWithLineSearch, SchurComplement) and `LinearSolver` (DenseLU).
- `equation_systems/` — `EquationSystem`, `LinearSystem`, `NonlinearSystem`, sparse/dense assembled vectors and matrices, axis layout.
- `dispatchers/` — work generators + schedulers (`SimpleScheduler`, `StaticHybridScheduler`, `SimpleMPIScheduler` when `NEML2_WORK_DISPATCHER=ON` brings in MPI) for splitting large batches across devices/processes.
- `drivers/` — `Driver`, `ModelDriver`, `TransientDriver` are the top-level "run a model over a load history" objects exposed in input files.
- `user_tensors/`, `misc/` — user-facing tensor input wrappers and odds-and-ends.

### Factory / Registry pattern

Every concrete `NEML2Object` (model, tensor, solver, driver, …) declares static `expected_options()` and self-registers via `register_NEML2_object*` macros. Input files then instantiate them by type name. Because each submodule is its own shared library, the umbrella header `include/neml2/neml2.h` declares `_neml2_force_link_*` C symbols and a `force_link_runtime()` helper that the executable / Python module must call to keep the linker from dropping the registration translation units. New submodules must add a `link_anchor.cxx` exposing such a symbol.

### Python bindings

`python/src/` contains the pybind11 module (`pyneml2`) with one `.cxx` per surface area (`core.cxx`, `tensors.cxx`, `crystallography.cxx`, `es.cxx`, `math.cxx`). The Python package in `python/neml2/` re-exports these (`from .core import *`, etc.) and adds pure-Python helpers: `pyzag/` (PyTorch-module interface to NEML2 models for training/AD), `reader/` (HIT-file reading, syntax helpers), `postprocessing/` (ODF, pole figures). `_cli.py` provides the console scripts.

## Conventions

- C++17, formatted by `clang-format` (enforced by `pre-commit` and CI). Run `pre-commit run --all-files` before pushing C++ changes.
- Python is formatted with `black` (line length 100, also CI-enforced).
- Headers live under `include/neml2/...`; sources under `src/neml2/...` with the same relative path. New headers without a corresponding test file will be flagged by reviewers.
- Notebooks under `python/examples/*.ipynb` are paired with MyST `.md` files via `jupytext --sync` (pre-commit hook); `.jupytext.toml` declares `formats = "ipynb,myst"`. Edit the `.ipynb`, never the paired `.md`.
- Avoid editing files in `contrib/`, `build/`, or `installed/` — those are generated. `scripts/clobber.sh [dir]` removes git-ignored files (with prompt) if a build gets wedged.
