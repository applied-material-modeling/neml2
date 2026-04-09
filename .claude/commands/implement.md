Implement a new NEML2 model class end-to-end.

`$ARGUMENTS` is a short description of what to implement (e.g. "traction-separation law", "CubicIsotropicHardening").

---

## Pipeline (execute in order — stop on failure)

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

---

### Step 4 — Write unit tests

Invoke the **test-writer agent** with the path to the new header.

The test-writer will:
1. Write `tests/unit/models/<path>/test_Foo.cxx` with `set_value` and `set_dvalue` sections
2. Use finite-difference derivative check at `atol=rtol=1e-5`
3. Confirm no CMakeLists.txt edit is needed (GLOB_RECURSE picks it up)

Then run `/build dev unit_tests` and `/test "Foo"` to verify the new test passes.

**If tests fail:** report the failure — do NOT auto-fix production code. Stop and ask the user.

**Only proceed to Step 5 when tests pass.**

---

### Step 5 — Remind

After Step 4, tell the user:

```
Next step:
  /docs-verify   — validate website output
```
