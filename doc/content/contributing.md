(contributing)=
# Contributing

A walkthrough of the development workflow for NEML2 contributors:
cloning the repository, getting the tests green, the linters and type
checkers wired into CI, the dependency-pinning machinery, and how to
build the docs locally.

If you are extending NEML2 with a new model, the
[](tutorials-extension) tutorials cover that side of the workflow
in depth.

## Source setup

Clone and install the package in editable mode with the dev extras:

```shell
git clone https://github.com/applied-material-modeling/neml2.git
cd neml2
pip install -e ".[dev]" -v
```

This drives [scikit-build-core](https://scikit-build-core.readthedocs.io/)
to compile the C++ extension and lays down the developer toolchain
(`pytest`, `ruff`, `pyright`, `pre-commit`, `sphinx`, …). The `-v` flag
streams the CMake / Ninja output, which is useful when something fails
to compile.

If you only need C++ source builds (no Python install), see the preset
table in [](build-customization).

## Pre-commit hooks

Install the hooks once per checkout:

```shell
pre-commit install
```

From then on, every `git commit` runs the configured hooks. The full
set (`.pre-commit-config.yaml`):

- `clang-format` — C++ formatter for files under `neml2/csrc/`.
- `ruff` (lint, `--fix`) and `ruff-format` — Python lint + formatter.
- `nmhit-format` — formatter for HIT input files (`*.i`).
- `jupytext --sync` — keeps notebook `*.ipynb` files and their paired
  `*.md` mirrors in lockstep (currently `doc/content/tutorials/optimization/{deterministic,statistical}/main.{ipynb,md}`).
- `check-notebook-executed` — fails if any code cell in a tracked
  `.ipynb` lacks an `execution_count`. Sphinx renders the committed
  outputs verbatim, so unexecuted cells would surface as blanks (or
  worse, stale outputs that no longer match the source). Re-run the
  notebook end-to-end before committing.
- `check-copyright` — adds or refreshes the MIT copyright header on
  source files.
- `check-dependencies` — verifies that every version string flagged
  with `# dependencies: <DEP>.<FIELD>` matches the canonical value in
  `dependencies.yaml` (see [](contributing-deps)).
- `check-compat-matrix` / `render-compat-matrix` — keep the torch
  compatibility table in `doc/content/installation/compatibility.md`
  in sync with `compatibility.yaml`.

Run the whole battery against the working tree at any time:

```shell
pre-commit run --all-files
```

(contributing-tests)=
## Tests

The test suite is `pytest`-driven and lives under `tests/`. Run the
full suite:

```shell
pytest -v tests/
```

For a single file (the `tests/` tree has five top-level buckets:
`unit/`, `models/`, `regression/`, `verification/`, `aoti/`):

```shell
pytest -v tests/unit/test_factory.py
```

The [pytest usage guide](https://docs.pytest.org/en/stable/how-to/usage.html)
documents the rich selector syntax (`-k expr`, `--lf`, parametrized
test IDs, marker filters, …).

The AOTI compile suite under `tests/aoti/` triggers an Inductor
compile per scenario (minutes per test) and runs by default; deselect
it with `pytest --deselect tests/aoti tests/` if you want the fast
subset for an inner edit loop.

VS Code users can drive the suite through the
[Python extension](https://github.com/Microsoft/vscode-python); set
`python.testing.pytestEnabled` to `true` and point `pytestArgs` at
`${workspaceFolder}/tests`.

## Linting and formatting

Both linters run automatically through pre-commit, but you can invoke
them directly:

```shell
ruff check .                 # lint (with auto-fix: `ruff check --fix .`)
ruff format .                # format
```

The `[tool.ruff]` section of `pyproject.toml` pins the enabled rule
families (`E`, `F`, `W`, `I`, `B`, `UP`) and configures `ruff format` as
the canonical Python formatter. The `lint` job in
`.github/workflows/python.yaml` gates every PR.

For C++ formatting, ensure your editor or IDE picks up the
`.clang-format` file at the repository root. The pre-commit hook covers
the same files (`*.cxx`, `*.h`) on commit.

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
`.github/workflows/python.yaml`); the editable shim resolves `import
neml2.*` to the in-tree source, so pyright always sees the same
sources the test suite does.

## Coverage

Python branch + line coverage is measured by `coverage.py` via the
`pytest-cov` plugin (both pinned in the `[dev]` extras). Configuration
lives in `pyproject.toml` under `[tool.coverage.run]` /
`[tool.coverage.report]`; the package source is `neml2/` and the
runner uses branch coverage on top of line coverage.

Local workflow:

```shell
pytest --cov tests/unit tests/models tests/aoti          # terminal summary
pytest --cov --cov-report=html tests/unit tests/models tests/aoti  # → htmlcov/index.html
pytest -n auto --cov tests/unit tests/models tests/aoti  # parallel (xdist-safe)
```

`tests/regression/` and `tests/verification/` are deliberately omitted
-- they pin gold outputs rather than exercise code paths, so including
them only inflates the coverage number with end-to-end exercises and
multiplies the run time. The `coverage` CI job in
`.github/workflows/python.yaml` runs the same subset with `-n auto`
and uploads the raw `coverage.xml` as a workflow artifact (30-day
retention).

The `fail_under` floor in `[tool.coverage.report]` starts at `0` -- a
real floor will be set once the first CI run establishes the baseline.

(contributing-deps)=

## Dependency pinning

Versions of third-party dependencies are tracked in `dependencies.yaml`
and propagated into source files (`pyproject.toml`,
`.github/workflows/*.yaml`, doc snippets, …) via inline annotations:

```python
# dependencies: torch.version_min
"torch>=2.10.0",
```

The annotation line above the version literal tells the
`scripts/dep_manager.py` tool which `dependencies.yaml` entry owns the
value. Use the tool rather than editing version strings by hand:

```shell
python scripts/dep_manager.py check                   # CI-equivalent verify
python scripts/dep_manager.py list                    # show all tracked deps
python scripts/dep_manager.py bump torch.version_min 2.11.0   # update + propagate
```

The torch compatibility matrix in `compatibility.yaml` is governed by
its own tool, `scripts/compat_matrix.py`. Both tools are wired into
pre-commit so a typo will be caught at commit time.

## Jupyter notebooks

Executable notebooks live next to the tutorial markdown that frames
them (currently
`doc/content/tutorials/optimization/{deterministic,statistical}/main.ipynb`).
Each notebook subfolder is self-contained: the notebook, its paired
markdown mirror, and any HIT input file it loads (e.g. the shared
`doc/content/tutorials/optimization/demo_model.i` referenced from both
calibration notebooks via `../demo_model.i`).

The paired-text workflow keeps notebooks reviewable and reproducible:

1. The `.ipynb` files are marked as binary in `.gitattributes` —
   their git diffs would otherwise be unreadable noise on metadata.
2. Each `.ipynb` is paired with a MyST markdown mirror via the
   `jupytext` pre-commit hook (`.jupytext.toml`). The `.md` is the
   review surface; PR diffs show meaningful cell-level changes.
3. **Edit the `.ipynb`, never the paired `.md`.** The hook regenerates
   the markdown.
4. After modifying a notebook, run all cells before committing — the
   `check-notebook-executed` pre-commit hook blocks commits with
   unexecuted code cells. Sphinx renders notebooks into the docs with
   pre-baked outputs (`nb_execution_mode = "off"` in `doc/conf.py`,
   no re-execution at build time), so a notebook with stale outputs
   ships stale outputs. The nightly `notebooks.yaml` workflow re-runs
   every notebook end-to-end and catches drift between PRs.

## Documentation

The doc pipeline is `sphinx-build` driven by `doc/conf.py`, wrapped by
`doc/scripts/build.sh`:

```shell
pip install -e ".[dev]" -v               # sphinx + extensions land here
doc/scripts/build.sh                     # build to doc/_build/html
xdg-open doc/_build/html/index.html      # macOS: `open ...`
```

The wrapper runs `sphinx-build -j auto -W --keep-going` with the
jupyter-cache pre-create workaround that lets parallel myst-nb workers
build without racing. `--help` lists every flag; the useful ones:
`--clean` for a cold rebuild, `--serve` to start `python -m http.server`
on `127.0.0.1:8765` (prints the SSH-tunnel command), `--port`,
`--dest`, `--no-strict`.

For live preview during editing:

```shell
sphinx-autobuild doc doc/_build/html
```

This serves at `http://127.0.0.1:8000/` and rebuilds whenever anything
under `doc/` changes.

The syntax catalog under `/generated/syntax/` is regenerated from
`neml2-syntax --json` on every build — see `doc/_ext/neml2_syntax.py`.

## Submitting a pull request

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
   the repository on GitHub, then branch off `main` in your fork.
2. Make your change, with tests where applicable.
3. Run `pre-commit run --all-files` and `pytest -v tests/` locally.
4. Push and open a PR against the upstream `main`. CI will run the
   same lint + test + typecheck matrix you ran locally, plus the
   wheel-build / torch-compat matrix defined in `.github/workflows/`.
5. Address review feedback by pushing new commits on top. Once a
   reviewer has started looking at the PR, avoid force-pushing —
   rewriting history makes it hard for them to see what changed
   between rounds. Squash / rebase happen at merge time.

## Use of generative AI

Contributions written with help from large language models or other
generative-AI tools are welcome. Two ground rules:

1. **Disclose.** Make it clear which parts of the change came from an
   AI tool. The two accepted forms are:

   - A `Co-authored-by:` trailer on the relevant commit(s) naming the
     tool — e.g. `Co-authored-by: Claude <noreply@anthropic.com>`.
   - A short note in the PR description identifying which files or
     sections were AI-assisted.

   Either is fine; pick the one that fits the granularity of the help.
   Boilerplate-level autocomplete (a few-token IDE suggestion) doesn't
   need to be called out.

2. **You own the result.** Whoever opens the PR — not the AI — is
   responsible for the merged change: that it is correct, that it
   follows project conventions, that the tests cover it, and that the
   license and provenance of any embedded snippets are clean. Treat
   AI output as draft code from an enthusiastic but unaccountable
   contributor: read it, test it, and rewrite the parts you wouldn't
   defend in review.

This policy is intentionally permissive. If the volume or quality of
AI-assisted PRs changes the calculus, we'll revisit.
