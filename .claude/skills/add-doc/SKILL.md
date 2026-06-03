---
name: add-doc
description: Add or revise a static NEML2 documentation page under `doc/content/` — physics-module pages, top-level reference pages, migration guides, and the cross-references between them. Use this for narrative documentation that doesn't need executable code cells. For runnable, notebook-style tutorial pages use the `add-tutorial` skill. For per-type catalog pages (`doc/generated/syntax/`), nothing — those autogenerate from `neml2-syntax --json` via the `neml2_syntax` Sphinx extension.
---

# add-doc

Static MyST reference pages — physics modules, top-level guides, the
migration set, anything that's prose + fenced code without
`{code-cell}` execution.

## Where to put it

| Page kind | Location | Toctree |
| :--- | :--- | :--- |
| Physics module (top-level) | `doc/content/modules/<name>.md` | `doc/content/modules/index.md` |
| Solid-mechanics sub-module | `doc/content/modules/solid_mechanics/<name>.md` | `doc/content/modules/solid_mechanics/index.md` |
| Top-level user-guide page | `doc/content/<name>.md` | `doc/index.md` (User guide caption) |
| Top-level Development page | `doc/content/<name>.md` | `doc/index.md` (Development caption) |
| Migration guide | `doc/content/migration/<from>_<to>.md` | `doc/content/migration/index.md` |

## Page shape

```
(label-anchor)=
# Title

One-paragraph orientation: what this page is for, who it's for.

## Math / physics motivation

$$ y = f(x; p) $$

## Worked input file

\```{literalinclude} ../../../tests/regression/<area>/<scenario>/model.i
:language: ini
:caption: tests/regression/<area>/<scenario>/model.i
\```

## Line-by-line explanation

- **`[block]`** ([](models-TypeName)) does X.
- **`[other]`** ([](models-OtherType)) does Y.

## See also

- [](tutorials-models-composition) for composition patterns.
- [](syntax-catalog) for the per-type option reference.
```

## Hard rules

- **No `{code-cell}`** on static pages. For executable content, use
  the `add-tutorial` skill.
- **HIT lexer is `ini`**, not `python`. The conf.py global
  `suppress_warnings = ["misc.highlighting_failure"]` quiets the
  fallback warnings, but `ini` is the canonical choice.
- **Cross-references** use MyST short form `[](label)`:
  - Syntax catalog: `models-<TypeName>`, `solvers-<Name>`,
    `drivers-<Name>`, `tensors-<Name>`, `data-<Name>`,
    `equationsystems-<Name>`.
  - Top-level pages: `getting-started`, `cli-utilities`,
    `aoti-packages`, `tensor-types`, `contributing`, `installation`,
    `tutorials`, `physics-modules`.
  - Sibling tutorials: `tutorials-models-<name>`,
    `tutorials-extension-<name>`, `tutorials-optimization-<name>`.
- **Real `{literalinclude}` paths.** Resolved relative to the source
  `.md`. `ls` the target before committing.
- **Add the new page to the parent `index.md` toctree.**
- **No legacy markup**: no `\note`, `\f$...\f$`, `{#anchor}`,
  `@list-input:`, `@ref`. Use `:::{note}`, `$...$` / `$$...$$`,
  `(anchor)=`, `{literalinclude}`, `[](anchor)`.

## Verify

```bash
sphinx-build -W -b html doc doc/_build/html
```

`-W` promotes broken xrefs / missing files / ambiguous anchors to
errors. The `build-docs` skill walks the full local-build pipeline.

## See also

- `add-tutorial` for notebook-style pages with `{code-cell}` blocks.
- `build-docs` for the local-build pipeline + failure-mode catalog.
- `doc/conf.py` for the active extension list.
