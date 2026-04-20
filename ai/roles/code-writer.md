# Skill: CODE-WRITER

Purpose: Implement or modify production C++ code only.

## Responsibilities

- headers under `include/neml2/`
- sources under `src/neml2/`
- minimal `expected_options()` docstrings
- build-system wiring only if explicitly required by non-glob CMake

## Rules

1. Do not write tests here.
2. Do not write full documentation here.
3. Add `register_NEML2_object(ClassName)` at the bottom of every new `.cxx`.
4. Write only one-line `.doc()` strings in `expected_options()` as a minimum.
5. When assigning from `Variable<T>`, use `operator()` to get the tensor value rather than copy-assigning the variable object.
6. Check whether derivative-disabled `.i` tests should be re-enabled after derivative fixes.

## Common C++ Type Pitfalls

- **`neml2::where(c, a, b)`**: All arguments MUST be NEML2 tensors (e.g., `Scalar`, `Vec`). No `double` or `bool` literals allowed. Use `Scalar(val, options)` or `Scalar::zeros/ones` matching the batch/intermediate dimensions.
- **Assignment to `Variable<T>`**: Use `_v = Tensor(t)` or `_v = t` if `t` is a NEML2 tensor. Avoid assigning components via `_v()(i) = ...`.
- **Casting**: Converting `at::Tensor` to `Scalar/Vec` requires the intermediate dimension: `Scalar(t, intmd_dim)`. For batch-only tensors, `intmd_dim` is usually `0`.
- **`Derivative` math**: `Derivative` objects often lack `operator-=`. Use `_t.d(_x) += -tensor` instead of `_t.d(_x) -= tensor`.
- **Default Parameters**: For `TensorName<T>` in `expected_options()`, set defaults as strings: `options.set<TensorName<Scalar>>("name") = "1.0";`.

## Workflow

1. Read similar nearby models first.
2. Implement the minimal correct code.
3. Check CMake wiring only after inspecting how sources are collected.
4. Hand off to [BUILD](../workflows/build.md), then [TEST-WRITER](./test-writer.md), then [DOC-WRITER](./doc-writer.md).
