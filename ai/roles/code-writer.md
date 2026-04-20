# Skill: CODE-WRITER

Purpose: Implement or modify production C++ code only.

## Responsibilities

- headers under `include/neml2/`
- sources under `src/neml2/`
- minimal `expected_options()` docstrings
- build-system wiring only if explicitly required by non-glob CMake

## Required Reading

Before writing or modifying NEML2 C++ code, read [NEML2-GUIDELINES](./neml2-guidelines.md).

## Rules

1. Do not write tests here.
2. Do not write full documentation here.
3. Add `register_NEML2_object(ClassName)` at the bottom of every new `.cxx`.
4. Write only one-line `.doc()` strings in `expected_options()` as a minimum.
5. When assigning from `Variable<T>`, use `operator()` to get the tensor value rather than copy-assigning the variable object.
6. Check whether derivative-disabled `.i` tests should be re-enabled after derivative fixes.

## Workflow

1. Read similar nearby models first.
2. Read [NEML2-GUIDELINES](./neml2-guidelines.md).
3. Implement the minimal correct code.
4. Check CMake wiring only after inspecting how sources are collected.
5. Hand off to [BUILD](../workflows/build.md), then [TEST-WRITER](./test-writer.md), then [DOC-WRITER](./doc-writer.md).
