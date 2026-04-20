# Role: NEML2-GUIDELINES

Purpose: Repository-specific C++ idioms and gotchas to read before editing NEML2 production code.

## When To Read This

Read this guide before writing or modifying C++ under `include/neml2/` or `src/neml2/`.

## Common Idioms And Gotchas

- **Variable Component Access**: Use `_var()(i)` when you need a component of the underlying tensor value. This distinguishes the `Variable<T>` wrapper from its contained tensor.
- **OptionSet Types**: In `expected_options()`, parameters usually use `TensorName<T>`, while inputs and outputs usually use `VariableName`. For primitive numeric options, prefer `double` or `int` rather than the `Real` alias, which may not be in scope.
- **Object Lifetime**: Do not bind `const Scalar &` members to temporary expressions in constructors, such as `_t - _t_old`. Store the owned value or bind only to an object with a stable lifetime.

## Common C++ Type Pitfalls

- **`neml2::where(c, a, b)`**: All arguments MUST be NEML2 tensors (e.g., `Scalar`, `Vec`). No `double` or `bool` literals allowed. Use `Scalar(val, options)` or `Scalar::zeros/ones` matching the batch/intermediate dimensions.
- **Assignment to `Variable<T>`**: Use `_v = Tensor(t)` or `_v = t` if `t` is a NEML2 tensor. Avoid assigning components via `_v()(i) = ...`.
- **Casting**: Converting `at::Tensor` to `Scalar/Vec` requires the intermediate dimension: `Scalar(t, intmd_dim)`. For batch-only tensors, `intmd_dim` is usually `0`.
- **`Derivative` math**: `Derivative` objects often lack `operator-=`. Use `_t.d(_x) += -tensor` instead of `_t.d(_x) -= tensor`.
- **Default Parameters**: For `TensorName<T>` in `expected_options()`, set defaults as strings: `options.set<TensorName<Scalar>>("name") = "1.0";`.
