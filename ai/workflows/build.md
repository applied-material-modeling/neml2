# Workflow: BUILD

Purpose: configure and build the C++ backend using a CMake preset.

## Inputs

- Optional preset, default `dev`
- Optional build target, default all
- Optional `--reconfigure`

## Valid Presets

- `dev`
- `release`
- `cc`
- `tsan`
- `asan`
- `coverage`
- `profiling`

## Procedure

1. If `build/<preset>/` does not exist, or `--reconfigure` is requested, run:
   - `cmake --preset <preset> -GNinja -S .`
2. Build:
   - `cmake --build --preset <preset> [--target <target>]`
3. If a header path from an error message is unclear, search for it under:
   - `include/neml2/**/<Filename>.h`
4. On success, report the output path.

## Light Repair Loop

Maximum 3 attempts total including the first build:

1. Extract the first meaningful compiler error.
2. If the error is trivial, limited to 1-2 files, apply the minimal fix and retry.
3. Trivial means cases such as:
   - missing `#include`
   - obvious typo
   - missing `register_NEML2_object(...)`
4. If the error is non-trivial, do not modify code. Report the file, line, and required fix.
5. If the exact same error repeats, stop immediately.
6. After 3 total attempts, stop and report the outstanding error.
7. If unresolved, recommend the [FIX-BUILD](./fix-build.md) procedure.

## Named Targets (Commonly used with `dev`)

- `unit_tests`
- `regression_tests`
- `verification_tests`
- `dispatcher_tests`
