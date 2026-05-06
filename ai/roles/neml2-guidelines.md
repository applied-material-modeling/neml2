# Role: NEML2-GUIDELINES

Purpose: Repository-specific C++ idioms and gotchas to read before editing NEML2 production code.

## When To Read This

Read this guide before writing or modifying C++ under `include/neml2/` or `src/neml2/`.

## Common Idioms And Gotchas

- **Variable Component Access**: Use `_var()(i)` when you need a component of the underlying tensor value. This distinguishes the `Variable<T>` wrapper from its contained tensor.
- **OptionSet API**: Use these helpers in `expected_options()`:
  - `options.add_input("name", "doc")` / `options.add_optional_input("name", "doc")` for input variables
  - `options.add_output("name", "doc")` / `options.add_optional_output("name", "doc")` for output variables
  - `options.add_parameter<T>("name", "doc")` for parameters; `options.add_buffer<T>("name", "doc")` for buffers
  - `options.add<T>("name", "doc")` for plain (non-tensor) options. Prefer `double`/`int` over the `Real` alias, which may not be in scope.
  - `options.set_private<bool>("define_second_derivatives", true)` to declare analytical second derivatives.
- **Variable Names**: Use bare names in both C++ (`declare_input_variable<T>("strain")`) and `.i` files — no subspace prefixes. For history (old-state) values use `history_name(name, /*nstep=*/N)`; for rates `rate_name(name)`; for residuals `residual_name(name)`. `VariableName` is just `std::string`.
- **Object Lifetime**: Do not bind `const Scalar &` members to temporary expressions in constructors, such as `_t - _t_old`. Store the owned value or bind only to an object with a stable lifetime.

## Common C++ Type Pitfalls

- **`neml2::where(c, a, b)`**: All arguments MUST be NEML2 tensors (e.g., `Scalar`, `Vec`). No `double` or `bool` literals allowed. Use `Scalar(val, options)` or `Scalar::zeros/ones` matching the batch/intermediate dimensions.
- **Assignment to `Variable<T>`**: Use `_v = Tensor(t)` or `_v = t` if `t` is a NEML2 tensor. Avoid assigning components via `_v()(i) = ...`.
- **Casting**: Converting `at::Tensor` to `Scalar/Vec` requires the intermediate dimension: `Scalar(t, intmd_dim)`. For batch-only tensors, `intmd_dim` is usually `0`.
- **`Derivative` math**: `Derivative` objects often lack `operator-=`. Use `_t.d(_x) += -tensor` instead of `_t.d(_x) -= tensor`.
