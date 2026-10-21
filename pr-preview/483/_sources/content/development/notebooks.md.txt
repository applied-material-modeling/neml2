# Jupyter notebooks

Every tutorial under `doc/content/tutorials/` is a self-contained
`main.ipynb` (plus any HIT input it writes for itself via a `%%writefile`
cell — there are no sibling files on disk). The notebooks are **not**
jupytext-paired: the `.ipynb` is the single source of truth, edited
directly and reviewed through GitHub's notebook diff.

There are two tiers, distinguished only by cost:

1. **Cheap tutorials** (most of them) are executed at build time
   (`nb_execution_mode = "cache"` in `doc/conf.py`) and committed
   **without outputs**. Sphinx runs each in a throwaway temp dir
   (`nb_execution_in_temp = True`) and caches the results under
   `_build/.jupyter_cache/`, so editing prose around a stable cell does
   not re-execute it.
2. **Expensive tutorials** — the two pyzag calibration notebooks,
   `optimization/{deterministic,statistical}/main.ipynb` — are listed in
   `nb_execution_excludepatterns` and committed **pre-executed**. Sphinx
   renders their committed outputs as-is (executing them needs a GPU and
   minutes-to-hours per run), and they get no "Open in Colab" badge.

Keeping notebooks reviewable and reproducible:

- **Edit the `.ipynb` directly.** GitHub renders cell-level diffs for
  notebooks in PRs. (`*.ipynb` is marked `binary` in `.gitattributes`,
  so local `git diff` shows "binary files differ" — review on GitHub or
  with `nbdime`.)
- **`ruff-format`** formats the code cells of every tracked `.ipynb` via
  the pre-commit hook.
- **Keep cheap notebooks output-free.** They are executed at build, so
  committed outputs would only add noise (and are not checked).
- **Keep the two expensive notebooks fully executed.** The scoped
  `check-notebook-executed` pre-commit hook blocks a commit if any of
  their code cells lacks an `execution_count` — an unexecuted cell would
  render empty or stale. The nightly `notebooks.yaml` workflow re-runs
  every notebook end-to-end and catches drift between PRs.
