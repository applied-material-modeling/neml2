Implement a new NEML2 model class end-to-end.

`$ARGUMENTS` is a short description of what to implement (e.g. "traction-separation law", "CubicIsotropicHardening").

---

## Pipeline (execute in order — stop on failure)

### Step 0 — Find design spec

Before writing any code, list all files under `design/` and find specs matching `$ARGUMENTS`
by module or model name. If `design/` does not exist, skip this step.

- **If one spec is found:** read it and use it as the source of truth. Report which file
  was used before proceeding.

- **If multiple specs are found in the same directory** (e.g. `design/traction_separation_law/BilinearMixed.md`,
  `design/traction_separation_law/SI.md`, `design/traction_separation_law/Linear.md`):
  read all of them. Implement each variant described by its own spec file.
  Decide whether a shared base class is appropriate based on the specs — if the variants
  share a common interface or governing structure, implement a base class first and have each
  variant inherit from it. Leave the inheritance design to judgment based on the specs.
  Report all spec files used and the chosen class hierarchy before proceeding.

- **If matching specs are spread across different directories:** choose the closest match by
  module subdirectory and model name, and implement only that one.

- **If no spec is found:** report "no design spec found under design/" and proceed using
  `$ARGUMENTS` and existing similar models as the implementation guide.

---

### Step 1 — Write the code

Invoke the **code-writer agent** with the description from `$ARGUMENTS`.

The code-writer will produce the header, `.cxx`, and `register_NEML2_object` registration.
It will NOT write tests or full documentation — those are handled in Steps 3 and 4.

---

### Step 2 — Build

Run `/build dev` and follow its repair loop.

**Only proceed to Step 3 when the build is clean.**
If the repair loop is exhausted and the build still fails, stop the pipeline and suggest
running `/fix-build` manually before retrying `/implement`.

---

### Step 3 — Complete documentation

Invoke the **doc-writer agent** with the path to the new `.cxx` file.

The doc-writer will:
1. Fill in complete `options.doc()` and per-option `.doc()` strings in `expected_options()`
2. Check neighboring headers and add a brief class comment to the header if local style warrants it
3. Locate the relevant narrative page under `doc/content/` (e.g. `doc/content/modules/solid_mechanics.md`)
   and add a section for the new model covering governing equations, variable definitions,
   parameter descriptions, and a minimal HIT input example using `@list-input`
4. If no suitable narrative page exists, create one and report where it should be linked

---

### Step 4 — Write unit tests

Invoke the **test-writer agent** with the path to the new header.

The test-writer will:
1. Write `tests/unit/models/<path>/test_Foo.cxx` with `set_value` and `set_dvalue` sections
2. Use finite-difference derivative check at `atol=rtol=1e-5`
3. Confirm no CMakeLists.txt edit is needed (GLOB_RECURSE picks it up)

Then run `/build dev unit_tests` and `/test "Foo"` to verify the new test passes.

**If tests fail — iterative fix loop (max 3 rounds):**

1. Diagnose the root cause of each failure:
   - **Test logic** (wrong expected value, bad input setup, stale reference data): fix the test directly and re-run.
   - **Production code** (crash in library code, wrong model output, assertion failure in `set_value`/`set_dvalue`): do NOT modify — stop immediately, report the exact file and line, and ask the user.
   - **Unclear**: do NOT modify anything. Explain your reasoning and ask the user before proceeding.

2. After each fix attempt, re-run the tests. If the same failure repeats unchanged, stop — do not loop further.

3. After 3 rounds of test-side fixes, if tests still fail, stop and report all outstanding failures to the user.

**Only proceed to Step 5 when tests pass.**

---

### Step 5 — Remind

After Step 4, tell the user:

```
Next step:
  /docs-verify   — validate website output
```
