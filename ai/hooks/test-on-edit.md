# Hook: Test-on-Edit

Purpose: Mandatory procedure after any C++ Source or Header edit.

## After Any C++ Source or Header Edit

If the edited file has a matching dedicated Catch2 test file, explicitly run the associated test.

### Procedure

1. If editing `test_Foo.cxx`, treat `Foo` as the source base.
2. Otherwise search `tests/unit/` for `test_<basename>.cxx`.
3. If no matching test file exists, stop this procedure silently.
4. If `build/dev/tests/unit/unit_tests` does not exist, note that tests cannot be rerun until a build exists.
5. Parse the first `TEST_CASE("...")` name from the test file.
6. Run:
   - `build/dev/tests/unit/unit_tests "<TestCaseName>"`

### Failure Policy

1. If the failure happens after editing source/header, first assume the test expectations may need updating.
2. Attempt test-file-only fixes for up to 3 rounds total.
3. Re-run the same test after each test-file fix.
4. If the same failure repeats unchanged, stop immediately.
5. If 3 test-side fix attempts fail, only then fix the source and explicitly report that source changes were required.
