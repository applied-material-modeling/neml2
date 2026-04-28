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
   - variable layout / store metadata checks
   - multi-step or driver workflows
4. Do not edit `CMakeLists.txt` if globbing already discovers new `.cxx` files.

## Derivative-Check Rules

1. `ModelUnitTest` already checks values, first derivatives, and second derivatives.
2. Do not write a hand-coded derivative section when `ModelUnitTest` is sufficient.
3. Any `check_values = false`, `check_derivatives = false`, or `check_second_derivatives = false` must have an explanatory comment immediately above it.
4. If a point is degenerate for forward finite differences, keep the check disabled there, explain why, and add a non-degenerate coverage case elsewhere.

## Test Scaffolding

- **Accuracy**: For complex models, do NOT calculate expected tensor values by hand. Write a temporary Python script (using `numpy` or `torch`) to implement the model's math and print the precise expected values for your `.i` file.
- **Tensors**: Use the `[Tensors]` section in `.i` files to define `Vec`, `SR2`, or `R2` inputs/outputs. Passing multi-component values as inline strings in `input_Vec_values` causes character-splitting parse errors.
- **Template**: Start from `ai/templates/SkeletonModelTest.i` — it contains correct `[Tensors]` structure, derivative flag comments, and the bare-name / `~N` history variable conventions.

### `[Tensors]` block patterns

```ini
# Scalar constant
[my_scalar]
  type = FullScalar
  value = 1.5
[]

# Vec from literal components (correct way — avoids inline splitting errors)
[my_vec]
  type = FillVec
  values = '1 0 0'
[]

# SR2 from scalar references
[my_sr2]
  type = FillSR2
  values = 'e11 e22 e33'    # each name must also be a [Tensors] entry
[]
```

### Derivative flags

| Flag | Meaning | Default recommendation |
|---|---|---|
| `check_values = true` | Verify forward output values | Always true |
| `check_derivatives = true` | FD check of `set_value` Jacobians | **Preferred default** — catches Jacobian bugs |
| `check_second_derivatives = true` | FD check of second derivatives | True if `define_second_derivatives = true` in model |
| `jit = false` | Skip TorchScript tracing | Use when debugging tracing errors |

If any check flag is `false`, add an explanatory comment immediately above it, e.g.:
```ini
# Jacobian is identically zero at this point; FD check would produce 0/0
check_derivatives = false
```

## Workflow

1. Read the target header.
2. Read a neighboring test in the same area.
3. Choose `.i` or `test_*.cxx` based on the behavior.
4. Write a complete test with no TODOs.
5. Build and run:
   ```
   cmake --build build/dev --target unit_tests
   ./build/dev/tests/unit/unit_tests "models"
   ```
