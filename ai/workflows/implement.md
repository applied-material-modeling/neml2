# Workflow: IMPLEMENT

Purpose: Implement a new NEML2 model end-to-end.

Execute these steps in order. Stop on failure.

---

## Step 0: Design Spec

1. Run `find design/ -type f` (or `ls`) to discover all spec files.
2. **Single match**: read it; use it as the source of truth.
   - **Jacobians**: If the spec provides analytical Jacobians, prioritize implementing them exactly as specified.
3. **Multiple specs in the same directory** (e.g. four files under `design/traction_separation_law/`):
   - Read ALL of them before writing any code.
   - Compare their **Input** and **Output** sections.
   - If every spec shares the same primary input type and the same primary output type
     (e.g. all receive `Vec displacement_jump` and all produce `Vec traction`), that shared
     interface IS sufficient justification for an abstract base class.
     Introduce the base class first, then implement each variant.
   - If the specs have divergent inputs or outputs, implement separate independent classes.
4. **Specs in different directories**: implement only the closest module/model match.
5. **No spec found**: proceed from the user request and similar existing models.

Before moving to Step 1, state:
- which spec files were used
- the planned class hierarchy (base + variants, or independent classes)

---

## Step 1: Production Code

Execute the [CODE-WRITER](../roles/code-writer.md) role guide.

**Jacobian Strategy**: 
- If analytical derivatives are complex, consider using NEML2's Automatic Differentiation (AD) initially (by calling `request_AD()` in the model's constructor) OR implement an explicit Jacobian and use AD as a temporary verification baseline.
- **Verification using AD**: To isolate derivation errors, implement the model logic in `set_value` without explicit derivatives first. Run tests with `check_derivatives = true` to get the ground truth from FD/AD. Then implement the analytical Jacobian and ensure it matches.

---

## Step 2: Build

Execute the [BUILD](./build.md) workflow with preset `dev`.

---

## Step 3: Tests

Execute the [TEST-WRITER](../roles/test-writer.md) role guide.

---

## Step 4: Iteration (if tests fail)

1. Fix test logic only when clearly appropriate (wrong expected value, wrong tolerance, wrong input setup).
2. If the failure is in production code, stop and report — do not silently change production logic here.
3. Stop after 3 rounds of test-side fixes if failures persist; report all outstanding failures.

---

## Step 5: Documentation

Execute the [DOC-WRITER](../roles/doc-writer.md) role guide.

---

## Step 6: Final Check

Remind the user to run or review [DOCS-VERIFY](./docs-verify.md).
