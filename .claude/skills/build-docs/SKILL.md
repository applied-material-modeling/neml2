---
name: build-docs
description: Build the NEML2 documentation locally. Use when the user asks to build, regenerate, preview, or serve the docs (e.g., "build the docs", "preview the documentation", "regenerate Python stubs", "the doc build is failing", "run examples.py", "run genhtml.py"). Walks through the `pip install` → `neml2-stub` → `doc/scripts/examples.py` → `doc/scripts/genhtml.py` pipeline and surfaces the gotchas (non-editable install requirement, unexecuted-cell failure mode, live-preview mode).
---

# build-docs

End-to-end documentation build for NEML2. The doc pipeline is two scripts under `doc/scripts/`, run on top of a freshly built/installed `neml2` Python package.

## Hard requirements before running

- The `neml2` Python package must be installed in **non-editable** mode in the active site-packages. An editable install (e.g., `pip install -e .`) will *not* work — the doc scripts import `neml2` and expect compiled artifacts in their final installed location. If the user has an editable install, they must `pip uninstall neml2` first and reinstall non-editable.
- All code cells in every notebook under `python/examples/*.ipynb` must already be executed. `doc/scripts/examples.py` checks for unexecuted cells and aborts. Re-execute the notebook in JupyterLab (or `jupyter nbconvert --to notebook --execute --inplace`) before re-running the build.
- Optional: `pre-commit install` so notebook/markdown pairs (`.ipynb` ↔ jupytext `.md`) stay in sync.

## Standard workflow

```bash
# 1. Install package + dev extras (non-editable!).
pip install ".[dev]" -v

# 2. Regenerate Python stubs whenever the pybind11 bindings have changed
#    (anything under python/src/*.cxx). Skip if bindings are unchanged.
neml2-stub

# 3. Run tutorials/examples to produce their outputs (figures, tables, …).
#    -GNinja is the standard; pass other CMake flags via --cmake-configure-args
#    if you need them (e.g., a custom CUDA toolchain).
./doc/scripts/examples.py --log-level INFO --cmake-configure-args="-GNinja"

# 4. Generate the HTML site.
./doc/scripts/genhtml.py --log-level INFO

# 5. Open the result.
xdg-open build/doc/build/html/index.html   # or `open` on macOS
```

## Live-preview mode (preferred during editing)

```bash
./doc/scripts/genhtml.py --serve
```

Serves at `http://127.0.0.1:8000/` and rebuilds automatically when tracked content changes. This skips the `examples.py` step — run that separately when notebook outputs need to be refreshed.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError` (or compiled-extension-not-found) when a doc script runs `import neml2` | Editable install of `neml2`, or no install at all | `pip uninstall neml2 && pip install ".[dev]" -v` (no `-e`) |
| `examples.py` aborts complaining about unexecuted cells in a notebook | Notebook was saved without running all cells | Open the notebook, run all cells, save, re-run `examples.py` |
| Stale Python signatures in the rendered docs | `neml2-stub` not re-run after binding changes under `python/src/*.cxx` | Re-run `neml2-stub`, then `./doc/scripts/genhtml.py` |
| Doxygen / syntax pages look out of date | `genhtml.py` step skipped or its log went unread | Re-run `./doc/scripts/genhtml.py --log-level INFO` and check the log |

## Reference

The canonical write-up of the documentation, testing, and contribution workflow is `doc/content/tutorials/contributing.md` (sections "Documentation" and "Jupyter notebooks"). When this skill and that document disagree, the `.md` file wins — update this skill to match.
