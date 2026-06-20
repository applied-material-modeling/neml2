(quality)=
# Code quality

## Pre-commit hooks

Install the hooks once per checkout:

```shell
pre-commit install
```

From then on, every `git commit` runs the configured hooks. The full
set (`.pre-commit-config.yaml`):

- `clang-format` — C++ formatter for `*.cxx`, `*.cpp`, and `*.h` files.
- `ruff` (lint with `--fix`, Python only) and `ruff-format` (Python plus
  notebook code cells) — Python lint + formatter.
- `jupytext --sync` — keeps tutorial notebook `*.ipynb` files and their
  paired `*.md` mirrors in lockstep.
- `check-notebook-executed` — fails if any code cell in a tracked
  `.ipynb` lacks an `execution_count`. Sphinx renders the committed
  outputs as-is, so re-run the notebook end-to-end before committing.
- `nmhit-format` — formatter for HIT input files (`*.i`).
- `check-copyright` — adds or refreshes the MIT copyright header on
  source files.
- `check-dependencies` — verifies that every version string flagged
  with `# dependencies: <DEP>.<FIELD>` matches the canonical value in
  `scripts/dependencies.yaml` (see [](contributing-deps)).
- `check-compat-matrix` / `render-compat-matrix` — keep the torch
  compatibility table in `doc/content/installation/compatibility.md`
  in sync with `doc/content/installation/compatibility.yaml`.
- `actionlint` — lints the GitHub Actions workflow files under
  `.github/workflows/`.

Run the whole battery against the working tree at any time:

```shell
pre-commit run --all-files
```

## Linting and formatting

Both linters run automatically through pre-commit, but you can invoke
them directly:

```shell
ruff check .                 # lint (with auto-fix: `ruff check --fix .`)
ruff format .                # format
```

The `[tool.ruff.lint]` section of `pyproject.toml` pins the enabled rule
families (`E`, `F`, `W`, `I`, `B`, `UP`) and `[tool.ruff]` configures
`ruff format` as the canonical Python formatter. The `lint` job in
`.github/workflows/python.yaml` gates every PR.

For C++ formatting, ensure your editor or IDE picks up the
`.clang-format` file at the repository root. The pre-commit hook covers
the same files (`*.cxx`, `*.cpp`, `*.h`) on commit.

## Type checking

Python type checking is performed by
[pyright](https://microsoft.github.io/pyright/). The default scope
(set in `pyproject.toml` `[tool.pyright]`) is `neml2`, `tests`,
`benchmark`, `scripts`, and `doc`. Run from the repository root:

```shell
pyright                      # check everything in scope
pyright neml2/types          # restrict to one subpath
```

CI runs `pyright` against the editable install (the `typecheck` job in
`.github/workflows/python.yaml`); the editable install resolves
`import neml2.*` to the in-tree source, so pyright generally sees the
same sources the test suite does. The compiled pybind extension is
resolved through auto-generated `.pyi` stubs co-located with their
`.so`; the `neml2-stub` console script regenerates them by running
`pybind11-stubgen` against the fully installed package, and CI invokes
it between `pip install` and `pyright`.
