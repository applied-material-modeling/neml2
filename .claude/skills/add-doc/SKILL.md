---
name: add-doc
description: Add or revise NEML2 user-facing documentation under `doc/content/` — physics-module pages, tutorial pages, system pages, and the cross-references between them. Use this skill whenever the user wants to write, edit, restructure, or audit narrative documentation (not the auto-generated syntax pages, which build themselves from `expected_options()` doc strings). Trigger on phrases like "document the new viscoelasticity domain", "add a tutorial page for X", "update the solid mechanics module doc", "the docs for Y are out of date", "register a new physics module page". Covers the editorial pattern (high-level intro + canonical names + pointer to syntax catalog), the `@list-input:` directive and its `git ls-files` quirk, and the `DoxygenLayout(Python).xml` registration step that's required when adding a brand-new top-level page.
---

# add-doc

Add or revise narrative documentation in `doc/content/`. Build is via `doc/scripts/genhtml.py` — see the `build-docs` skill for the build pipeline.

This skill is about **what to write and where**, not the build itself. Two recurring failure modes drive its existence:

1. Per-class deep dives that go stale because the auto-generated syntax pages already cover the same ground (with maintenance burden) — the editorial guidance below explains the alternative.
2. New top-level pages that render fine in isolation but never appear in the navigation sidebar because both `DoxygenLayout.xml` and `DoxygenLayoutPython.xml` were not updated.

## Where docs live

```
doc/
├── content/
│   ├── modules/                    Per-physics-module pages (the "what physics can NEML2 model")
│   │   ├── solid_mechanics.md
│   │   ├── chemical_reactions.md
│   │   ├── phase_field_fracture.md
│   │   └── …
│   ├── system/                     Per-system pages (Model, Driver, Solver, Tensor, …)
│   ├── tutorials/                  Step-by-step tutorial pages, often with paired notebooks
│   └── DoxygenLayoutPython.xml     Wait, see below — this is in doc/config/
└── config/
    ├── DoxygenLayout.xml           Sidebar/navigation for the C++ docs
    └── DoxygenLayoutPython.xml     Sidebar/navigation for the Python docs (mirrors DoxygenLayout.xml)
```

The auto-generated **syntax pages** (`syntax-models.html`, `syntax-tensors.html`, etc.) are produced by `extract_syntax` from the installed `neml2` Python package's registry — they are *not* in `doc/content/` and you cannot edit them directly. They reflect every registered object's `expected_options()` doc strings; that is where per-class detail belongs, not in narrative prose.

## Choosing where a change lands

| You're doing | Where it goes | Layout XML edit needed? |
|---|---|---|
| Adding a section to an existing physics module (e.g., a new domain inside `solid_mechanics`) | New `##` section inside the existing `doc/content/modules/<submodule>.md` | No |
| Adding a brand-new top-level physics module | New file `doc/content/modules/<submodule>.md` | **Yes — both XMLs** |
| Adding a tutorial page | New file under `doc/content/tutorials/<topic>/<page>.md` | **Yes — both XMLs**, registered as a `<tab type="user">` under the relevant `usergroup` |
| Polishing per-class documentation | Edit the C++ class's `options.doc()` and `expected_options()` strings — the syntax page rebuilds from those | No |

## Editorial pattern: high-level + canonical names + pointer to the catalog

The strongest temptation when documenting a new domain is to write a `###` subsection per class, with each class's full equations and parameters. **Resist this.** Per-class detail is already in the syntax pages, generated automatically from `expected_options()`. Mirroring it in narrative prose creates a maintenance burden — every new model in the domain forces a doc edit, and the prose drifts out of sync with the registry over time.

The pattern that ages well is:

1. **One paragraph of theory.** What this class of models describes physically; what distinguishes it from neighbouring categories.
2. **One paragraph of framing.** How NEML2 represents this category: which primitive elements/operators are involved, how internal state is handled (rates + time integration + implicit solve, or pure forward evaluation), what the typical composition pattern looks like.
3. **A name list with a pointer to the syntax catalog.** Mention 3–6 canonical pre-assembled objects by name (textbook configurations the reader will recognize), and link to `[Syntax Documentation](@ref syntax-models)` for the full list.
4. **One end-to-end input file example.** Pick the most representative scenario — usually the one that exercises the typical composition pattern (`ImplicitUpdate` + `NonlinearSystem` + `Newton`). Embed via `@list-input:` from a regression test (see below).

Example structure:

```markdown
## <Domain title>

<Paragraph 1: physical theory and what distinguishes this category.>

<Paragraph 2: how NEML2 frames it — primitive elements, time integration, composition pattern.>

Standard textbook configurations are pre-assembled as named objects — for example `Foo`, `Bar`, `Baz`, … Refer to [Syntax Documentation](@ref syntax-models) for the complete catalog, including each object's parameters, inputs, and outputs.

The example input file below shows the typical composition: <one-sentence orientation>.

@list-input:tests/regression/<submodule>/<scenario>/model.i:Models,EquationSystems,Solvers
```

The same pattern applies to *existing* domain docs that have grown a per-class subsection per model — those are candidates for compaction whenever you touch them.

## The `@list-input:` directive

`@list-input:<path>:<sections>` embeds a slice of an existing `.i` file directly into the rendered docs, so the input-file examples track the test files automatically. Do not write a separate hand-edited input snippet:

- `@list-input:path/to/file.i:Models` — only the `[Models]` block.
- `@list-input:path/to/file.i:Models,EquationSystems,Solvers` — those three blocks (typical for a complete `ImplicitUpdate` example).
- `@list-input:path/to/file.i:Models/foo,Models/bar` — only sub-blocks `[foo]` and `[bar]` inside `[Models]`. Use this when one regression file backs several doc snippets that focus on different sub-models.

**`@list-input:` resolves paths via `git ls-files`, not the working tree.** A referenced file that is untracked will fail at build time with `no files found for '<path>'`. Run `git add` on the new test files *before* building the docs — staging is enough; you don't need to commit. This is the most common cause of a doc-build failure when adding new content, because the test files you point at were typically just created in the same change.

Prefer regression-test `model.i` files for examples that involve time integration (they show the full composition), and unit-test `.i` files for one-off "this is what this object alone looks like" snippets.

## Cross-referencing

- Link to a Doxygen anchor: `[Friendly title](@ref anchor-name)`. Anchors come from the `{#anchor-name}` suffix on a top-level Markdown heading (`# Solid Mechanics {#solid-mechanics}`) or from the page's filename.
- Common targets: `[Syntax Documentation](@ref syntax-models)`, `[Syntax Documentation](@ref syntax-tensors)`, `[Syntax Documentation](@ref syntax-drivers)`, `[Syntax Documentation](@ref syntax-solvers)`. The `system/*.md` pages each link to the right syntax page — mirror that wording.
- Wrap object names in `` `Backticks` `` (renders as `<span class="tt">…</span>`). Don't add a manual link to the syntax page on every name — Doxygen does not autolink object names from the registry into the syntax pages, and link-spam crowds the prose.

## Adding a brand-new top-level page

If the new content gets its own file under `doc/content/modules/`, `doc/content/tutorials/`, or `doc/content/system/`, you must register it in **both** layout XMLs:

- `doc/config/DoxygenLayout.xml` — uses `@ref <anchor>` URLs (Doxygen-internal references).
- `doc/config/DoxygenLayoutPython.xml` — mirrors the same structure, but uses relative `../<page>` URLs to point back into the C++ docs.

Both files are entirely declarative — find the right `<tab type="usergroup">` (e.g., `Module Documentation`, `Guides and Tutorials`) and add a `<tab type="user" url="@ref new-page" title="New Page Title"/>` entry in the appropriate spot. **The two files must mirror each other**; if you add an entry to one and forget the other, the page will appear in only one of the two doc trees.

A new `## section` *inside* an existing top-level page (the common case for adding a domain to `solid_mechanics.md`) does **not** require any XML edit — Doxygen builds the in-page table of contents automatically.

## Verifying the build

Use the `build-docs` skill for the full pipeline. Two gotchas worth knowing in advance:

- **Force re-preprocessing when only Markdown changed.** `genhtml.py` caches its preprocess pass; if you re-run after editing only the `.md` (or after staging files that the previous run failed to find), it may report `no markdown updates detected; skipping preprocess step` and reuse the old HTML. Touch the modified `.md` file (`touch doc/content/...`) to force a fresh preprocess.
- **Syntax-catalog cross-references rely on the *installed* package, not the dev tree.** `extract_syntax` reads the registry from the Python package in `site-packages`. Newly registered classes won't appear on the syntax page (or be cross-referenceable) until the Python package is reinstalled (`pip install ".[dev]" -v`). The narrative page itself will build fine — but its `@ref syntax-models` link will land on a page that doesn't yet list the new objects.

## Notebook-paired tutorials

Tutorial pages under `doc/content/tutorials/` are often paired with executable Jupyter notebooks under `python/examples/<tutorial-name>/*.ipynb`. The `.ipynb` and a Markdown mirror are kept in sync by a `jupytext` pre-commit hook (configured by `.jupytext.toml`, `formats = "ipynb,myst"`). **Always edit the `.ipynb`, never the paired `.md`** — the hook will overwrite a hand-edited `.md` on commit. The `examples.py` pipeline step also requires every code cell to have been executed; an unexecuted cell will abort the doc build.

## When in doubt

The canonical write-up of the documentation, testing, and contribution workflow is `doc/content/tutorials/contributing.md`. When this skill and that document disagree, the `.md` file wins — update this skill to match.
