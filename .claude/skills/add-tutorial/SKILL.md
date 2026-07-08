---
name: add-tutorial
description: Author a new NEML2 tutorial page under `doc/content/tutorials/`. Tutorials are self-contained, Colab-runnable Jupyter notebooks (`main.ipynb`, no jupytext pairing) with executable code cells; they create their own input files in-notebook via `%%writefile` (no sibling files, no `{literalinclude}`) and carry an auto-injected "Open in Colab" badge. Trigger on phrases like "add a tutorial for X", "write a tutorial about Y", "I need a notebook-style doc page demonstrating Z". For static reference pages without executable cells, see `add-doc` instead.
---

# add-tutorial

A tutorial page renders runnable Python inline next to its captured
output — `myst-nb` extracts `{code-cell} ipython3` blocks, executes them
at build time, and weaves the output into the HTML. The *same* notebook
also opens in **Google Colab** (every tutorial gets an auto-injected
"Open in Colab" badge), so it must be **self-contained**: it installs
NEML2 and writes out any input files it needs, rather than relying on
sibling files that only exist in the repo.

Tutorials sit under `doc/content/tutorials/{models,extension,
optimization}/<name>/` as a directory containing a single
self-contained `main.ipynb`.

## Voice (most important rule)

Tutorials are **user-facing**. The reader showed up to *do* something,
not to learn how NEML2 is built. Write to a working engineer or
materials scientist who just installed the package — friendly,
concrete, and economical with jargon. Specific rules:

- **Open with what the reader will accomplish in one or two
  sentences**, not with a formal model definition.
- **Lead with running code, then explain.** Most readers run the cells
  first and read prose only when something surprises them. Prose
  *before* a code cell is for context they need to interpret the
  code, not for theory dumps.
- **Domain math good; meta-abstraction bad.** Equations that describe
  the material model itself — `σ = 3K·vol(ε) + 2G·dev(ε)`, a hardening
  curve, a flow rule — add clarity and belong in the tutorial. They
  tell the reader what the code is actually computing. Drop the
  *abstractions over NEML2 itself*: things like the `y = f(x; p, b)`
  framing, generic claims about "input variables, output variables,
  parameters, and buffers" before the reader has seen one, or wrapping
  a concrete example in unnecessary symbology. Show the physics with
  symbols; skip the meta-notation that merely restates the function
  signature.
- **Skip implementation jargon.** Avoid "declarative surface of a
  Model subclass", "trainable parameters that participate in
  autograd", "the forward operator", "broadcast-state tangents".
  These read as insider language. Say "your model needs an input" /
  "parameters can be fitted to data later" / "the function NEML2
  will call".
- **Don't enumerate every option.** "Young's modulus and Poisson's
  ratio" is enough on first encounter; the reference catalog lists
  the other parameterisations for anyone who wants them. Link, don't
  list.
- **One concept per tutorial.** If the reader has to track inputs,
  outputs, parameters, AND buffers in tutorial 1, they're
  overwhelmed. Split. Cross-link the follow-ups via `[](label)`.
- **Skip "abstractly" detours.** If a paragraph starts with "more
  abstractly" or "in general", delete it — that's reference-page
  material.
- **The reader doesn't care about your refactor history.** No "v2
  used to do X; v3 now does Y" framing inside tutorials. (That's
  migration-guide territory.)

A simple test before merging: a colleague who's never opened NEML2
should be able to follow the page top-to-bottom without opening
another tab. If they have to, the tutorial leaned on unexplained
jargon or assumed a missing prerequisite.

## Layout

```
doc/content/tutorials/<section>/<name>/
└── main.ipynb   — the rendered + Colab-launched notebook (the only file)
```

**No sibling files.** Anything the notebook loads (`input.i`,
`projectile.py`, …) is created *inside* the notebook with a
`%%writefile` cell — there is nothing on disk next to `main.ipynb` for
Colab to miss. Do **not** add an `input.i` file or use
`{literalinclude}` (the directive does not render in Colab).

Author `main.ipynb` directly — it is a real Jupyter notebook (no
jupytext pairing, no `.md` mirror). Draft it in Jupyter, or construct it
with `nbformat`; the cell-by-cell template below shows what goes in each
cell. Commit it with **no outputs** (the docs build executes the
notebook; only the two expensive optimization notebooks are committed
pre-executed). `ruff-format` formats the `.ipynb` code cells via the
pre-commit hook.

## Notebook metadata

A real `.ipynb` needs only a valid `kernelspec` in its notebook
metadata — myst-nb reads the code cells natively, so there is no
front-matter to add:

```json
"metadata": {
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 }
}
```

Do **not** set a per-notebook execution mode
(`metadata.mystnb.execution_mode`) — execution is controlled globally in
`doc/conf.py` (`nb_execution_mode = "cache"`).

## Body structure

The template below shows the notebook's cells in order — a markdown cell
for each prose block, a code cell for each `{code-cell}`. It is written
in MyST-markdown form for readability; the committed artifact is the
equivalent `.ipynb`.

````markdown
(tutorials-<section>-<name>)=
# Title

One or two sentences on what the reader will accomplish here. Link
prerequisites with [](label) only if they're truly required to follow
along.

```{code-cell} ipython3
:tags: [remove-cell]

# Install NEML2 when running in Google Colab; no-op in the docs build and
# local Jupyter, and hidden from the rendered page.
import sys

if "google.colab" in sys.modules:
    !pip install -q neml2
```

## The input file

Run the cell below to write this tutorial's input file:

```{code-cell} ipython3
%%writefile input.i
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

## Loading the model

```{code-cell} ipython3
import neml2
model = neml2.load_model("input.i", "<model name>")
model
```

## Evaluating

```{code-cell} ipython3
import torch
from neml2.types import SR2
strain = SR2(torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64))
model(strain)
```

## (Optional) The math behind it

If the equation IS the lesson, put it here, after the reader has
already seen the model run. Math in MyST: `$...$` inline, `$$...$$`
display, `\begin{align}` for numbered groups. If the math just
restates the code, drop this section.

## Where to go next

- [](next-tutorial-anchor) — what to read after this.
````

## Hard rules

- **Bootstrap cell first.** The hidden `:tags: [remove-cell]` pip cell
  is the first cell, before any `import neml2`. It is a no-op outside
  Colab and invisible in the rendered docs.
- **`%%writefile` must be the first line of its cell** — no comment or
  blank line before it (it's a cell magic). Everything after it becomes
  the file's contents, so write the whole file (e.g. a complete
  `projectile.py`, not one method).
- **Self-contained, no siblings.** Executed cells run in a throwaway
  temp dir (`nb_execution_in_temp = True`), so a `%%writefile input.i`
  cell must run *before* the `load_model("input.i", ...)` that reads it;
  both resolve against that temp cwd. Never assume a file exists on disk.
- **Code cells** run under the Python 3 kernel. (In the MyST template
  above they appear as `{code-cell} ipython3` fences.)
- **Cross-references**: `[](label)` MyST short form. Common labels:
  - Tutorials: `tutorials-models-<name>`,
    `tutorials-extension-<name>`, `tutorials-optimization-<name>`.
  - Syntax catalog (autogenerated): `models-<TypeName>`,
    `solvers-<Name>`, `drivers-<Name>`.
  - Top-level pages: `getting-started`, `cli-utilities`,
    `aoti-packages`, `tensor-types`, `contributing`.
- **Shell commands**: use `!cmd ...` shell magic inside a `{code-cell}`
  block (e.g. `!neml2-compile input.i --model elasticity`), *not*
  `{program-output}` (it runs in a different phase than myst-nb
  execution) and *not* `subprocess.run` wrappers.
- **The "Open in Colab" badge is automatic** — `doc/_ext/colab_button.py`
  injects it from the page's `.ipynb` path. Don't add a badge by hand.
- **Pre-verify the notebook is self-contained**: execute it in an empty
  temp dir before committing —
  `d=$(mktemp -d); cp main.ipynb "$d"; (cd "$d" && python -m jupyter nbconvert --to notebook --execute main.ipynb --output o.ipynb)`.
  It must run with no siblings present. (Or build the docs.)
- **Toctree wiring**: add `<section>/<name>/main` to the corresponding
  `index.md` toctree. Sphinx renders the `.ipynb` directly.

## Execution model

The global `nb_execution_mode` is `"cache"` and `nb_execution_in_temp`
is `True`: the first build executes each tutorial in a temp dir and
reuses cached outputs (`_build/.jupyter_cache/`) when the cell source is
unchanged. The two **expensive** pyzag notebooks
(`optimization/{deterministic,statistical}`) are the exception — they are
listed in `nb_execution_excludepatterns`, committed pre-executed, kept
that way by the scoped `check-notebook-executed` hook, and get no Colab
badge.

The live docs site deploys on **release** (not every push to main), so a
tutorial's Colab badge points at the released tag and the `pip install
neml2` it runs matches the deployed page.

## See also

- The `build-docs` skill for the local-build pipeline + the front-
  matter gotcha catalog.
- `add-doc` for static reference pages (no code cells).
- Exemplar tutorial:
  `doc/content/tutorials/models/running_your_first_model/main.ipynb`.
