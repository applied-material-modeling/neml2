# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NEML2 is a Python-native material modeling library that vectorizes constitutive model evaluation on CPU/GPU using PyTorch as the tensor backend. Models are plain `torch.nn.Module` subclasses composed from small reusable pieces, the framework auto-resolves dependencies between them, and most users interact via HIT input files (the same format used by MOOSE) plus either the Python API or the `neml2-run` / `neml2-compile` CLIs.

The legacy C++ tower from v2.x was retired in the v3 migration. The only C++ that remains is `neml2/csrc/aoti/`, a thin runtime that loads the AOT-Inductor `.pt2` packages produced by `neml2-compile`; the runtime is built once into `neml2/lib/libneml2_aoti.so` as part of the wheel and is invisible to most contributors. See [](doc/content/migration/212_300.md) for the v2 → v3 rewrite summary.

## Build & develop

```bash
pip install -e ".[dev]" -v       # editable + dev extras (pytest, pre-commit, sphinx, ...)
```

This drives a `scikit-build-core` build of the small AOTI C++ runtime under `build/<wheel_tag>/`, then installs the Python package in editable mode. Python source edits take effect immediately; touching anything under `neml2/csrc/` or `CMakeLists.txt` requires a re-`pip install` to rebuild the runtime.

Package versions and pinned deps live in `dependencies.yaml` — use `python scripts/dep_manager.py {check|list|bump DEP.FIELD VALUE}` rather than editing version strings by hand. Files reference their dep with a `# dependencies: NAME.FIELD` annotation immediately above the version literal. The torch compatibility matrix (`compatibility.yaml`) is a separate registry checked against the same `dependencies.yaml` torch entry — keep them in sync via `python scripts/compat_matrix.py {seed|check|render}`.

### C++ AOTI runtime (rarely needed)

```bash
cmake --preset dev               # configure (Debug)
cmake --build --preset dev       # build libneml2_aoti
cmake --preset cc                # configure-only; exports compile_commands.json for clangd
```

Only two presets exist: `dev` (build) and `cc` (compile-commands-only, no `cmake --build`). The wheel build that `pip install` drives uses `cmake.build-type = "Release"` internally — there is no developer-facing `release` preset.

## Tests

All tests are pytest. Test layout under `tests/` is organised by subsystem:

```bash
pytest tests/                                  # everything
pytest tests/test_model.py                     # one file
pytest tests/test_model.py::test_forward       # one function
pytest tests/regression/                       # the parametrized regression sweep
pytest tests/verification/                     # the parametrized verification sweep
pytest --run-aoti-compile tests/aoti/          # opt in to the slow AOTI compile tests
```

`tests/regression/test_regression.py` and `tests/verification/test_verification.py` discover scenarios by walking their respective directories for `.i` files and emit one parametrize id per scenario (the input file's relative path). For spot-checks after a code change, target a single scenario via `-k` or the full parametrize id:

```bash
pytest tests/regression/test_regression.py -k maxwell
pytest 'tests/regression/test_regression.py::test_regression[solid_mechanics/viscoelasticity/maxwell/model.i]'
```

Reserve unfiltered `tests/regression/` / `tests/verification/` runs for final confirmation.

When adding a `Model` subclass, use the `/add-model` skill; for a regression or verification scenario use `/add-regression` or `/add-verification`. The skills encode the test conventions so you don't have to rediscover them.

## Documentation

Sphinx with the shibuya theme and MyST-NB. Build with:

```bash
sphinx-build -j auto -W --keep-going -b html doc doc/_build/html
```

The `/build-docs` skill walks through the full pipeline (including notebook execution caching and the auto-generated HIT-syntax catalog). The `neml2` package must be importable for autodoc / `neml2-syntax` to introspect the registered objects — an editable `pip install -e ".[dev]"` is enough.

## CLI tools

The installed wheel exposes four console scripts (defined in `pyproject.toml` under `[project.scripts]`):

- `neml2-run <input.i>` — drive a model through a load history.
- `neml2-inspect <input.i>` — print the resolved input/output graph of a wired-up input file. Use this *before* `neml2-run` when composing models; wiring bugs surface as obvious mismatches instead of cryptic shape errors deep in Newton.
- `neml2-syntax --section Models --summary` — browse the registered-object catalog with one-line docstrings (`--type <Name>` to drill into one). Run this when planning *any* new Model or wondering whether a primitive already does what you want.
- `neml2-compile <input.i> --model <name>` — export a model to an AOT-Inductor `.pt2` package + drop-in HIT stub. See [](doc/content/model_compilation/) for the artifact format and pipeline reference.

`neml2-diagnose` and `neml2-time` from v2 are gone.

## Architecture

The Python package layout under `neml2/`:

- `model.py` — `Model` base class (a `torch.nn.Module`). All user-authored constitutive leaves inherit from this.
- `factory.py` — HIT input parsing via `nmhit`, `load_input` / `load_model` / `load_nonlinear_system` entry points, `[Tensors]` namespace for inline Python expressions.
- `schema.py` — declarative HIT syntax: `input`, `output`, `parameter`, `option` field helpers + `HitSchema` container that drives both parsing and the auto-generated docs.
- `chain_rule.py` — type aliases for the first / second-order chain-rule sensitivity dicts threaded through `Model.forward(..., v=, v2=, vh=)`.
- `resolver.py` — `DependencyResolver` builds the `ComposedModel` dependency graph from individual leaves' declared inputs and outputs.
- `models/` — the composable forward operators. `ComposedModel` glues children together via the dependency graph; `ImplicitUpdate` wraps a residual model in a Newton solve with optional `Predictor`. Domain libraries live in `models/{solid_mechanics,chemical_reactions,phase_field_fracture,porous_flow,finite_volume,common,kwn}/`; crystal plasticity is a subdirectory of `solid_mechanics/`.
- `types/` — typed tensor wrappers (`Scalar`, `Vec`, `R2`, `SR2`, `Rot`, `Quaternion`, `MillerIndex`, fourth-order `SSR4` / `WSR4` / ...). Each is a dataclass registered with `torch.utils._pytree.register_dataclass` so it round-trips through `torch.export`. `.data` exposes the underlying `torch.Tensor`.
- `solvers.py` — `NonlinearSolver` (Newton, NewtonWithLineSearch, SchurComplement) and `LinearSolver` (DenseLU).
- `equation_systems.py` — `EquationSystem`, `LinearSystem`, `NonlinearSystem`, sparse/dense assembled vectors and matrices, axis layout. Dense/Block variants (`DenseRHS`, `DenseNewtonStep`, `DenseIFT`, `BlockRHS`, ...) back both the eager Newton loop and the AOTI implicit-segment lowering.
- `drivers/` — `TransientDriver`, `ModelUnitTest`, `TransientRegression`, `Verification` — the top-level "run a model over a load history" objects exposed in input files.
- `data/` — `CubicCrystal`, `CrystalGeometry` and related crystallography data classes.
- `user_tensors/` — registered `[Tensors]` block types other than `Python` (currently the `CSV<Type>` family).
- `export.py` — adapter around `torch.export` + `torch._inductor.aoti_compile_and_package`; the single entry point through which every AOTI lowering passes.
- `cli/aoti_compile.py`, `cli/aoti_export.py` — the `neml2-compile` orchestration and the per-segment export path; see [](doc/content/model_compilation/pipeline.md).
- `aoti/` — Python-side `AOTIModel` shim that loads a `_meta.json` + per-segment `.pt2` files and exposes `forward` / `jvp` / `jacobian`. Backed by the pybind module `aoti/_aoti.cxx` (which links `libneml2_aoti.so`).
- `pyzag/` — `NEML2PyzagModel` adapter that exposes a NEML2 `Model` as a `torch.nn.Module` consumable by the pyzag training library.
- `cli/` — backing modules for the four console scripts above.
- `csrc/aoti/` — C++ runtime: `neml2::aoti::Model` wraps `torch::inductor::AOTIModelPackageLoader`. Built into `neml2/lib/libneml2_aoti.so` and surfaced through the pybind binding.

### Factory / Registry pattern

Every concrete native object (model, driver, tensor, solver, ...) self-registers via the `@register_neml2_object("TypeName")` decorator from `neml2.factory`. HIT input files then instantiate them by type name. `Model` subclasses declare their input/output/parameter surface via a class-level `hit = HitSchema(...)` and a `from_hit` constructor; the `@register_neml2_object` decorator + `HitSchema` together feed both the live factory and the auto-generated `neml2-syntax` catalog.

When adding a new submodule under `neml2/models/<domain>/`, append the import to the parent `__init__.py` so `import neml2` triggers registration — there is no lazy-loading machinery and unimported modules' types are invisible to the factory.

## Conventions

- Python source: linted and formatted with `ruff` (line length 100), CI-enforced via the `lint` job in `.github/workflows/python.yml`. Run `pre-commit run --all-files` before pushing.
- Type-checked with `pyright` against the installed package (CI: the `typecheck` job).
- Math in docstrings uses MyST dollarmath (`$x$` inline, `$$...$$` display); MyST `dollarmath` and `amsmath` extensions are enabled in `doc/conf.py`. Code references stay in `` `backticks` ``; the difference matters for rendering in the syntax catalog.
- HIT inputs use the `nmhit` Python parser (also a pre-commit `nmhit-format` hook). The format itself is unchanged from v2.
- Notebooks under `doc/content/tutorials/**.ipynb` are paired with `.md` mirrors via `jupytext --sync` (pre-commit hook); `.jupytext.toml` declares the format pairing. Edit the `.ipynb`, never the paired `.md`.
- Copyright headers are checked by a pre-commit hook (`python scripts/check_copyright.py`); the script auto-fixes missing headers.
- Avoid editing files in `build/`, `installed/`, or `doc/_build/`, `doc/generated/` — those are generated. `scripts/clobber.sh [dir]` removes git-ignored files if a build gets wedged.
