# Role: DOC-WRITER

Purpose: Write and update documentation only — not production code, not tests.

---

## Documentation Targets

### 1. Option docstrings (`expected_options()`)

- Fill in missing or weak `.doc()` strings in every `.cxx` touched.
- One-line strings only; describe what the parameter controls.
- This is the minimum viable output for any task.

### 2. Header comments

- Add only if the local header style already uses them.
- Read at least 3 neighboring headers before deciding.
- One-line `/** @brief ... */` maximum; never duplicate formulation text here.

### 3. Narrative documentation (`doc/content/`) — mandatory for new models

When the task introduces a new model, constitutive law, or physics object, `doc/content/` **must** be updated. This is not optional.

**Find the right page:**
- New solid mechanics model → `doc/content/modules/solid_mechanics.md`
- New physics domain → check for an existing page via `Glob: doc/content/modules/*.md`, or create one

**Add a section** (or create a new page if none fits), following the conventions of surrounding content:
- Use LaTeX `\f[ ... \f]` for block equations, `\f$ ... \f$` inline
- List all input/output variables (bare names, no subspace prefix) and their physical meaning
- Include a minimal HIT input example using the `@list-input` directive:
  ```
  @list-input:tests/unit/models/<path>/ModelName.i:Models
  ```
  (use the unit test `.i` file if no dedicated example exists yet)

For new constitutive laws, the section must cover:
- governing equations
- variable definitions and sign conventions
- parameter descriptions and units
- any special behavior (piecewise, unloading/reloading, irreversibility, compression cutoff)
- the `@list-input` HIT example

**If creating a new page:**
- Place it under `doc/content/modules/<domain>.md`
- Report the intended link location (which navigation file or index page should reference it)

### 4. Python docstrings

Use Google style with `Args:`, `Returns:`, and `Raises:` when applicable.

---

## Workflow

1. Read every `.cxx` touched by this task.
2. Fill in missing or weak `.doc()` strings (→ Target 1).
3. Check local header style; add a brief comment only if warranted (→ Target 2).
4. For new models: locate or create the `doc/content/modules/` page and write the required section (→ Target 3). **Do not skip this step.**
5. Run [DOCS-VERIFY](../workflows/docs-verify.md).
