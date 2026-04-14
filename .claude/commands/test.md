Run NEML2 tests (C++ and/or Python).

Parse `$ARGUMENTS` to decide what to run. Default: all unit tests + all Python tests.

---

## Preset resolution

`$ARGUMENTS` may begin with a known preset name (`dev`, `release`, `cc`, `tsan`, `asan`, `coverage`, `profiling`). If present, strip it from `$ARGUMENTS` and use it as `<preset>`. Otherwise default to `dev`.

**Auto-detect fallback** — if `build/<preset>/tests/unit/unit_tests` does not exist:
1. Search for the binary in any other `build/*/tests/unit/unit_tests`.
2. If exactly one is found, use that preset and note it to the user.
3. If multiple are found, pick the most recently modified one and note it.
4. If none are found, prompt the user to run `/build` first.

---

## Argument dispatch

| Argument                        | Action                                                      |
|---------------------------------|-------------------------------------------------------------|
| _(none)_                        | All C++ unit tests + all Python tests                       |
| `cpp` or `c++`                  | All four C++ test suites                                    |
| `python` or `py`                | Python tests via pytest                                     |
| `unit`                          | C++ unit tests only                                         |
| `regression`                    | C++ regression tests only                                   |
| `verification`                  | C++ verification tests only                                 |
| `dispatcher`                    | C++ dispatcher tests only                                   |
| `[tag]` (starts with `[`)       | Unit tests filtered by Catch2 tag, e.g. `[tensors]`         |
| Any other string                | Treated as a Catch2 test case name or pytest node ID        |

A preset word at the start of `$ARGUMENTS` is consumed before dispatch. Examples:
- `/test release [tensors]` — run unit tests tagged `[tensors]` from the `release` build
- `/test dev regression` — run regression suite from the `dev` build

---

## C++ commands

Replace `<preset>` with the resolved preset name (see above).

```bash
# All unit tests
./build/<preset>/tests/unit/unit_tests

# Filter by Catch2 tag (folder path under tests/unit/)
./build/<preset>/tests/unit/unit_tests "[tensors]"
./build/<preset>/tests/unit/unit_tests "[models/solid_mechanics]"

# Filter by test case name
./build/<preset>/tests/unit/unit_tests "LabeledAxis"

# Other suites
./build/<preset>/tests/regression/regression_tests
./build/<preset>/tests/verification/verification_tests
./build/<preset>/tests/dispatchers/dispatcher_tests
```

## Python commands

```bash
# All Python tests
pytest -v python/tests

# Single test file or function
pytest -v python/tests/test_foo.py
pytest -v python/tests/test_foo.py::test_bar
```

---

**After running, report:** total tests / passed / failed / elapsed time / names of failing tests with their error messages. Use the same format for all suites (unit, regression, verification, dispatcher).

---

## Failure triage

For each failing test, determine the root cause before acting:

| Root cause | Evidence | Action |
|---|---|---|
| **Test logic** | Wrong expected value, bad input setup, stale reference data | Fix the test directly |
| **Production code** | Assertion fails on correct logic, crash in library code, wrong model output | State the file + lines, describe the fix — do NOT modify |
| **Unclear** | Could be either | Do NOT modify anything. Explain your reasoning and ask for confirmation before proceeding. |

When the fix is in production code, always check for consistency across:
- `include/neml2/...` (header declaration)
- `src/neml2/...` (implementation)
- `tests/unit/...` (test expectations)
- Relevant `CMakeLists.txt`
