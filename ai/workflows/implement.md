# Workflow: IMPLEMENT

Purpose: Implement a new NEML2 model end-to-end.

Execute these steps in order. Stop on failure unless an iteration rule below explicitly allows a
small local loop.

---

## Step 0: Design Spec

Specs may be `.md`, `.pdf`, `.txt`, `.html`, or `.htm`. Markdown is preferred. Convert complex
non-Markdown specs to Markdown before code edits.

### Decision Tree

**A. User provides spec files** — Use them before searching `design/`.

1. Read all provided files before code edits.
2. Unsupported extension: stop and ask for `.md`, `.pdf`, `.txt`, `.html`, or `.htm`.
3. Markdown is source of truth.
4. Non-Markdown:
   - straightforward: implement from it
   - not straightforward: generate a Markdown spec under `design/`, then implement from that

**B. No provided spec, `design/` has specs** — Search recursively for `.md`, `.pdf`, `.txt`,
`.html`, `.htm`.

Relevant Markdown wins over non-Markdown.

1. **Single match**: read it; use it as the source of truth.
   - **Jacobians**: If the spec provides analytical Jacobians, implement them exactly as specified.
   - If non-Markdown and not straightforward, generate Markdown first.
2. **Multiple specs in the same directory**:
   - Read ALL of them before writing any code.
   - Treat the directory as a candidate model family.
   - Convert non-straightforward non-Markdown specs before comparing interfaces.
   - Compare their **Input** and **Output** sections.
   - If every spec shares the same primary input type and the same primary output type
     (e.g. all receive `Vec displacement_jump` and all produce `Vec traction`), that shared
     interface IS sufficient justification for an abstract base class.
     Introduce the base class first, then implement each variant in that directory as part of the
     same implementation scope by default.
   - If the specs have divergent inputs or outputs, implement separate independent classes.
3. **Specs in different directories**: implement only the closest module/model match.

**C. No spec found, request is straightforward** — single model, clear math, unambiguous interface:

- Proceed directly from the user prompt and the nearest existing similar model.
- No spec file needs to be created.

**D. No spec found, request is complex** — multiple variants, ambiguous interface, non-trivial math,
or a model family:

- Stop before writing code.
- Ask the user to place a supported spec file under `design/`.
- Show the [Spec Format](#spec-format) and ask for `Input`, `Algorithm`, and `Output`.

### Spec Format

When asking for or generating Markdown, show/use this concise format:

- `# Input`: current inputs, state inputs, parameters, controls; include type, units, component order
- `# Algorithm`: equations, branches, guards, state updates, implementation notes
- `# Output`: primary outputs, updated state outputs, optional derived outputs
- Optional: `## Jacobians`, `## Source files read`, `## Open issues`

`Input`, `Algorithm`, and `Output` are mandatory. Use bare NEML2 variable names unless quoting
legacy source names.

### Scope Rule (applies to all cases)

Scope narrowing is opt-in, never implicit. If multiple same-directory specs or a complex request implies a full model family, the default scope is the FULL family. Do not silently stop after the easiest variant. If you intend to defer any variant, explicitly mark it `DEFERRED` and get user approval before writing code.

Before moving to Step 1, state:
- which spec files were used, including any generated Markdown spec, or that none was found and the
  request is straightforward
- the planned class hierarchy (base + variants, or independent classes)
- the implementation scope for this turn: full family or an explicitly named subset/MVP
- the exact variant list covered by this scope, naming every spec-derived class/model explicitly
- any deferred variant, explicitly marked as `DEFERRED`, with a user-approved reason; otherwise there must be no deferred variants

---

## Step 1: Discovery

Before writing any code, orient yourself in the local physics domain.

1. **Find the reference model.** In the same subdirectory where the new model will live, identify the simplest complete existing model (e.g. `LinearIsotropicHardening` in `solid_mechanics/`, `GreenLagrangeStrain` in `solid_mechanics/elasticity/`). Read its `.h` and `.cxx`.
2. **Confirm local conventions.** Note: which base class is used, the bare variable names used for inputs/outputs (e.g. `equivalent_plastic_strain`; history variables use the `~N` suffix), and how parameters are declared (`add_parameter<T>(...)` + `declare_parameter<T>(...)`).
3. **Confirm CMake collection.** Run `grep -rn "GLOB\|glob_recurse" src/neml2/models/$(dirname of new model)/CMakeLists.txt` to verify sources are auto-collected. If auto-collected, no CMake edit is needed.
4. **Load the template.** Read `ai/templates/SkeletonModel.h` and `ai/templates/SkeletonModel.cxx` for structural reference.

Before moving to Step 2, state:
- Which reference model was read and its file path
- The parent class that will be used
- The bare variable names for all inputs and outputs (and any `~N` history names)
- Whether CMake collection is by glob (no edit needed) or explicit (edit needed)

---

## Step 2: Production Code

Execute the [CODE-WRITER](../roles/code-writer.md) role guide.
Read [NEML2-GUIDELINES](../roles/neml2-guidelines.md) before any NEML2 C++ edit in this step.

**Execution Model**:
- Treat production code, targeted build checks, and test-fixture setup as a controlled small-loop
  workflow, not a strict one-way handoff.
- In this loop, CODE-WRITER owns production files and TEST-WRITER owns test files.
- You may move between code edits, targeted builds, and test-input authoring when each pass is
  directly informing the same feature.
- Keep the loop narrow: prefer 1-3 edited production files plus the immediately related tests.
- Restrict the loop to one feature and one responsibility boundary at a time. Do not use the active
  loop to expand into adjacent features, neighboring model families, or unrelated infrastructure
  cleanup. (Steps 2-4 define this loop.)
- If Step 0 identified a same-directory model family as the active feature, that family is the
  feature boundary. Do not claim completion while any declared variant remains unimplemented.
- Do not use "co-evolution" to blur responsibility. If a failure is clearly caused by production
  logic, fix production logic. If a failure is clearly caused by test setup or expected values, fix
  only the test.
- When a test tolerance or fixture is adjusted to support iteration, state why that adjustment is
  numerically or procedurally justified.
- If test edits start changing unrelated expectations, broadening shared fixtures, or altering
  coverage outside the active feature boundary, stop and report before continuing.

**Jacobian Strategy**: 
- Default to implementing the production Jacobian explicitly. Do not treat `request_AD()` as a peer implementation path unless the user or spec explicitly asks for an AD-backed model.
- **Use AD only as a verification aid**: when derivative derivation is error-prone, first make `set_value` correct, enable derivative checks to obtain the FD/AD reference, then implement the analytical Jacobian and confirm it matches before considering the model complete.

---

## Step 3: Build

Execute the [BUILD](./build.md) workflow with preset `dev`.

**During Step 2-4 Iteration**:
- You may run targeted builds before Step 3 is "formally complete" when they are needed to validate
  a local production edit.
- Prefer `cmake --build --preset dev --target unit_tests` or a narrower target over a full build
  when that is sufficient to validate the active loop.
- BUILD light repair is limited to the active feature boundary and must obey Step 2 scope rules.

---

## Step 4: Tests

Execute the [TEST-WRITER](../roles/test-writer.md) role guide.
Start from `ai/templates/SkeletonModelTest.i` for the fixture structure.

**During Step 2-4 Iteration**:
- New declarative `ModelUnitTest` inputs may be authored before production code is fully stable if
  they are being used to lock down the intended interface, expected values, or derivative behavior.
- Prefer running the narrowest test that covers the active change. Broaden only after the targeted
  test passes.

---

## Step 5: Iteration (if tests fail)

This step owns failure repair inside IMPLEMENT. The TEST workflow and Test-on-Edit hook do not
override this policy.

1. Fix test logic only when clearly appropriate (wrong expected value, wrong tolerance, wrong input setup).
2. If the failure is in production code, fix the production code when it is still part of the same active implementation loop.
3. If the failure is non-trivial, ambiguous, or would require broadening scope beyond the active loop, stop and report instead of guessing.
4. If the next plausible fix would touch unrelated tests, shared fixtures, or neighboring features, stop and report instead of silently widening scope.
5. Stop after 3 rounds of test-side fixes if failures persist; report all outstanding failures.

**When all tests pass: proceed immediately to Step 6. Tests passing is not the end of IMPLEMENT.
Do not stop, do not summarize, do not wait. Steps 6 and 7 are mandatory.**

After edits, run applicable hooks: [Post-Edit](../hooks/post-edit.md) for formatted files and
[Test-on-Edit](../hooks/test-on-edit.md) after C++ source/header or model test edits.

---

## Step 6: Documentation

Execute the [DOC-WRITER](../roles/doc-writer.md) role guide.

---

## Step 7: Final Check

Report the [DOCS-VERIFY](./docs-verify.md) result. If it was skipped or failed, state why and what
remains.
