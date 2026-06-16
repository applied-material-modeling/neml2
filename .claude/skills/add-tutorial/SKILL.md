---
name: add-tutorial
description: Author a new NEML2 tutorial page under `doc/content/tutorials/`. Tutorial pages are MyST-NB documents with executable `{code-cell}` blocks, paired with a sibling `input.i` shown via `{literalinclude}`. Trigger on phrases like "add a tutorial for X", "write a tutorial about Y", "I need a notebook-style doc page demonstrating Z". For static reference pages without executable cells, see `add-doc` instead.
---

# add-tutorial

A tutorial page renders runnable Python inline next to its captured
output — `myst-nb` extracts `{code-cell} ipython3` blocks into a
notebook, executes them at build time, and weaves the output into the
HTML. Tutorials sit under `doc/content/tutorials/{models,extension,
optimization}/<name>/` as a directory containing `main.md` plus any
supporting `input.i` / sibling files.

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
├── main.md       — narrative + code cells + literalincludes
└── input.i       — the HIT input the code cells load (optional)
```

Single-page tutorials without supporting files live as a flat
`<name>.md` (the `optimization/pyzag.md` pattern).

## Required front-matter (mandatory)

Without the `jupytext` block, myst-nb silently drops `{code-cell}`
directives and the page renders the prose only.

```yaml
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: cache
---
```

## Body structure

```markdown
(tutorials-<section>-<name>)=
# Title

One or two sentences on what the reader will accomplish here. Link
prerequisites with [](label) only if they're truly required to follow
along.

## The input file

\```{literalinclude} input.i
:language: ini
:caption: input.i
\```

Brief prose if the input file uses anything non-obvious. If it's a
basic block, the file speaks for itself.

## Loading the model

\```{code-cell} ipython3
import neml2
model = neml2.load_model("input.i", "<model name>")
model
\```

## Evaluating

\```{code-cell} ipython3
import torch
from neml2.types import SR2
strain = SR2(torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64))
model(strain)
\```

## (Optional) The math behind it

If the equation IS the lesson, put it here, after the reader has
already seen the model run. Math in MyST: `$...$` for inline, `$$...$$`
for display, `\begin{align}` for numbered groups. If the math just
restates what the code does, drop this section.

## Where to go next

- [](next-tutorial-anchor) — what to read after this.
```

## Hard rules

- **Code cells**: triple-backtick \`\`\`{code-cell} ipython3\`\`\` form,
  language **must be `ipython3`** (`python` parses but doesn't
  execute).
- **HIT code blocks**: language `ini` (`python` triggers Pygments
  warnings on `!include` and `${var}`).
- **Cwd for executed cells** is the source file's directory, so
  `load_model("input.i", ...)` finds an `input.i` sibling without
  path gymnastics.
- **Cross-references**: `[](label)` MyST short form. Common labels:
  - Tutorials: `tutorials-models-<name>`,
    `tutorials-extension-<name>`, `tutorials-optimization-<name>`.
  - Syntax catalog (autogenerated): `models-<TypeName>`,
    `solvers-<Name>`, `drivers-<Name>`.
  - Top-level pages: `getting-started`, `cli-utilities`,
    `aoti-packages`, `tensor-types`, `contributing`.
- **Shell commands** in a tutorial: use `!cmd ...` shell magic inside
  a `{code-cell}` block, *not* `{program-output}`. Sphinx-side
  `{program-output}` runs during the docutils transform phase, before
  myst-nb executes code cells — artifacts produced by one won't be
  visible to the other.
- **No `subprocess.run` wrappers** for CLI demos. `!neml2-compile ...`
  inside a code cell is cleaner and renders as a shell session.
- **Pre-verify every cell** standalone (`python -c "..."`) before
  pasting in. Build-time failures cost a full rebuild round-trip.
- **Toctree wiring**: add the new tutorial to the corresponding
  `index.md` toctree in the parent directory.

## Cache + per-page opt-in

The global `nb_execution_mode` is `"off"` so plain markdown stays
fast. The `mystnb.execution_mode: cache` front-matter opts the
tutorial in — first build executes, subsequent builds reuse cached
outputs via `_build/.jupyter_cache/` when the cell source is
unchanged.

## See also

- The `build-docs` skill for the local-build pipeline + the front-
  matter gotcha catalog.
- `add-doc` for static reference pages (no code cells).
- Exemplar tutorial:
  `doc/content/tutorials/models/running_your_first_model/main.md`.
