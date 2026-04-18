# Skill: TEST-WRITER

Purpose: Write tests only.

## Decision Order

1. Read the model header.
2. Read the surrounding test directory.
3. Match the nearest existing local pattern.

## Rules for `Model` Subclasses

1. Default to declarative `.i` tests under `tests/unit/models/`.
2. Use `ModelUnitTest` unless procedural logic cannot be expressed declaratively.
3. Write a dedicated `test_*.cxx` only for procedural behaviors such as:
   - exception handling
   - parser/factory wiring checks
   - axis metadata checks
   - multi-step or driver workflows
4. Do not edit `CMakeLists.txt` if globbing already discovers new `.cxx` files.

## Derivative-Check Rules

1. `ModelUnitTest` already checks values, first derivatives, and second derivatives.
2. Do not write a hand-coded derivative section when `ModelUnitTest` is sufficient.
3. Any `check_values = false`, `check_derivatives = false`, or `check_second_derivatives = false` must have an explanatory comment immediately above it.
4. If a point is degenerate for forward finite differences, keep the check disabled there, explain why, and add a non-degenerate coverage case elsewhere.

## Workflow

1. Read the target header.
2. Read a neighboring test in the same area.
3. Choose `.i` or `test_*.cxx` based on the behavior.
4. Write a complete test with no TODOs.
5. Build and run the targeted test.
