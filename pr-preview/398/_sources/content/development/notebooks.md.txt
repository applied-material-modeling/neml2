# Jupyter notebooks

Executable notebooks live next to the tutorial markdown that frames
them (currently
`doc/content/tutorials/optimization/{deterministic,statistical}/main.ipynb`).
Each notebook subfolder is self-contained: the notebook, its paired
markdown mirror, and any HIT input file it loads (e.g. the
`demo_model.i` referenced from each calibration notebook).

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
   pre-baked outputs (no re-execution at build time), so a notebook
   with stale outputs ships stale outputs. The nightly `notebooks.yaml`
   workflow re-runs every notebook end-to-end and catches drift between
   PRs.
