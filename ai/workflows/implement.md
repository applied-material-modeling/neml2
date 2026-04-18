# Workflow: IMPLEMENT

Purpose: Implement a new NEML2 model end-to-end.

Execute these steps in order. Stop on failure.

## Step 0: Design Spec
1. Search `design/` for matching `.md` or `.pdf` specs.
2. If one spec matches, use it as the source of truth.
3. If multiple specs in the same directory match, read all of them and implement each variant.
4. If variants share a common structure, introduce a shared base class only if justified by the specs.
5. If matching specs span different directories, choose the closest module and model match.
6. If no spec exists, proceed using the user request and similar existing models.

## Step 1: Production Code
Execute the [CODE-WRITER](../skills/code-writer.md) skill.

## Step 2: Build
Execute the [BUILD](./build.md) workflow with preset `dev`.

## Step 3: Tests
Execute the [TEST-WRITER](../skills/test-writer.md) skill.

## Step 4: Verification
1. `cmake --build --preset dev --target unit_tests`
2. run the most relevant test or tests

## Step 5: Iteration (if tests fail)
1. fix test logic only when clearly appropriate
2. if failure is in production code, stop and report rather than silently changing production logic here
3. stop if the same failure repeats unchanged
4. stop after 3 rounds of test-side fixes

## Step 6: Documentation
Execute the [DOC-WRITER](../skills/doc-writer.md) skill.

## Step 7: Final Check
Remind the user to run or review [DOCS-VERIFY](./docs-verify.md).
