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
- `jupytext --sync` — keeps `notebooks/*.ipynb` and their paired
  `*.md` mirrors in lockstep.
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

For a single file:

```shell
pytest -v tests/test_model.py
```

The [pytest usage guide](https://docs.pytest.org/en/stable/how-to/usage.html)
documents the rich selector syntax (`-k expr`, `--lf`, parametrized
test IDs, marker filters, …).

Some tests are gated behind opt-in markers:

- `--run-aoti-compile` — also exercise the AOTI export path (default
  off because each test compiles a model, which is slow).

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
`.github/workflows/python.yml` gates every PR.

For C++ formatting, ensure your editor or IDE picks up the
`.clang-format` file at the repository root. The pre-commit hook covers
the same files (`*.cxx`, `*.h`) on commit.

## Type checking

Python type checking is performed by
[pyright](https://microsoft.github.io/pyright/). The default scope
excludes the `neml2` package itself, so contributors can run pyright
without first compiling:

```shell
pyright                      # default scope: scripts, doc, tests
pyright neml2                # also type-check the package (requires the build)
```

CI runs pyright in both scopes — the package-scope job installs the
wheel and runs against `import neml2` resolved from `site-packages`.

(contributing-deps)=

## Dependency pinning

Versions of third-party dependencies are tracked in `dependencies.yaml`
and propagated into source files (`pyproject.toml`,
`.github/workflows/*.yml`, doc snippets, …) via inline annotations:

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

Notebooks under `notebooks/` use a paired-text workflow so their
substance is reviewable and reproducible:

1. The `.ipynb` files are marked as binary in `.gitattributes` —
   their git diffs would otherwise be unreadable noise on metadata.
2. Each `.ipynb` is paired with a MyST markdown mirror via the
   `jupytext` pre-commit hook (`.jupytext.toml`). The `.md` is the
   review surface; PR diffs show meaningful cell-level changes.
3. **Edit the `.ipynb`, never the paired `.md`.** The hook regenerates
   the markdown.
4. After modifying a notebook, run all cells before committing.
   Notebooks are rendered into the docs with pre-baked outputs (no
   re-execution at build time), so a notebook with stale outputs ships
   stale outputs.

## Documentation

The doc pipeline is `sphinx-build` driven by `doc/conf.py`:

```shell
pip install -e ".[dev]" -v   # sphinx + extensions land here
sphinx-build -b html doc doc/_build/html
xdg-open doc/_build/html/index.html      # macOS: `open ...`
```

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
