# Hook: Test-on-Edit

Purpose: Mandatory procedure after any C++ Source or Header edit.

## After Any C++ Source or Header Edit

Inside IMPLEMENT, this hook is subordinate to IMPLEMENT Step 5 failure repair.
If the edited file has a matching dedicated Catch2 test file, explicitly run the associated test.
If the edit introduces or materially changes declarative model tests under `tests/unit/models/**/*.i`,
run the model suite entry point instead of relying only on basename matching.

### Procedure

1. If editing `test_Foo.cxx`, treat `Foo` as the source base.
2. Otherwise search `tests/unit/` for `test_<basename>.cxx`.
3. Also inspect whether the current change added or updated any declarative tests under `tests/unit/models/**/*.i`.
4. If declarative model tests were added or changed, first choose the narrowest available scope:
   - if all changed `.i` files lie under one subdirectory such as `tests/unit/models/solid_mechanics/elasticity/`, prefer a sub-area command or future tag/filter dedicated to that subtree
   - if the repo does not yet provide a narrower runnable filter for that subtree, record that limitation and fall back to the model-suite entry point
5. The fallback command for declarative model tests is:
   - `build/dev/tests/unit/unit_tests "models"`
6. If neither a matching Catch2 file nor declarative model test exists, stop this procedure silently.
7. If `build/dev/tests/unit/unit_tests` does not exist, note that tests cannot be rerun until a build exists.
8. If using a matching Catch2 file, parse the first `TEST_CASE("...")` name from the test file.
9. Run:
   - `build/dev/tests/unit/unit_tests "<TestCaseName>"`
   - or the narrowest declarative-model command identified in steps 4-5

### Failure Policy

1. If the failure happens after editing source/header, first assume the test expectations may need updating.
2. Attempt test-file-only fixes for up to 3 rounds total.
3. Re-run the same targeted command after each test-file fix.
4. If the same failure repeats unchanged, stop immediately.
5. If 3 test-side fix attempts fail, only then fix the source and explicitly report that source changes were required.
