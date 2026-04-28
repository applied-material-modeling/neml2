# Workflow: TEST

Purpose: Run C++ and/or Python tests.

## Input Dispatch

- no argument: all C++ unit tests and all Python tests
- `cpp` or `c++`: all four C++ suites
- `python` or `py`: Python tests only
- `unit`: C++ unit tests only
- `regression`: regression suite only
- `verification`: verification suite only
- `dispatcher`: dispatcher suite only
- `[tag]`: Catch2 tag filter
- any other string: Catch2 test name or pytest node id

## Preset Resolution

1. If the first argument is a known preset, use it and remove it from the remaining test argument.
2. Otherwise default to `dev`.
3. If `build/<preset>/tests/unit/unit_tests` is missing:
   - search other `build/*/tests/unit/unit_tests`
   - if one is found, use it
   - if several are found, use the most recently modified one
   - if none are found, stop and report that a build is required first

## C++ Commands

- unit: `./build/<preset>/tests/unit/unit_tests`
- tag: `./build/<preset>/tests/unit/unit_tests "[tag]"`
- named test: `./build/<preset>/tests/unit/unit_tests "TestName"`
- regression: `./build/<preset>/tests/regression/regression_tests`
- verification: `./build/<preset>/tests/verification/verification_tests`
- dispatcher: `./build/<preset>/tests/dispatchers/dispatcher_tests`

## Python Environment Resolution

1. Find a Python interpreter that imports both `torch` and `neml2`.
2. If none exists, find one that imports `torch`.
3. If `torch` exists but `neml2` does not, run [SETUP](./setup.md) once, then retry interpreter discovery.
4. If an interpreter exists but lacks `pytest`, install `pytest`.
5. If no interpreter with `torch` exists, stop and report that `torch` must be installed manually.
6. Do not auto-install `torch`.

## Python Commands

- full suite: `<py> -m pytest -v python/tests`
- single file or node: `<py> -m pytest -v python/tests/test_foo.py`
- single function: `<py> -m pytest -v python/tests/test_foo.py::test_bar`

## Reporting

Always report totals, passed, failed, elapsed time, and failing test names with error messages.

## Failure Triage

1. If the problem is clearly test logic, fix the test.
2. If the problem is clearly production code, report the file and lines; do not modify production code during test triage.
3. If unclear, stop and report the ambiguity instead of guessing.

When production-code consistency matters, inspect all relevant locations:
- `include/neml2/...`
- `src/neml2/...`
- `tests/unit/...`
- relevant `CMakeLists.txt`
