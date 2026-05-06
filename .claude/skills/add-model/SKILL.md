---
name: add-model
description: Scaffold a new NEML2 `: public Model` subclass — header, registered source, and `.i` ModelUnitTest input. Use this skill whenever the user wants to add, create, or scaffold any Model-family class: constitutive models (yield functions, hardening laws, flow rules, elasticity, eigenstrains), or non-constitutive Model subclasses (converters, parameter maps, smoothing functions, geometric helpers). Trigger on phrases like "add a new isotropic hardening model", "create a model that maps X to Y", "scaffold a YieldFunction subclass", "add a Foo model under solid_mechanics", even when the user doesn't say the word "Model" — if it derives from `Model`, this skill applies.
---

# add-model

Scaffold a new `: public Model` subclass in NEML2 — header, source with the registration macro, and a `ModelUnitTest` input file. CMake auto-discovers all three via `file(GLOB_RECURSE)`, so no CMake edits are needed when adding under an existing submodule.

## When to use

Use this skill for any class that derives from `Model`: constitutive models, but also converters, parameter maps, smoothing functions, geometric helpers — anything in `include/neml2/models/`. Typical phrasing: "add a new model that …", "scaffold a Foo model under solid_mechanics", "create a YieldFunction subclass".

Do **not** use this skill for:
- Adding a new tensor class (under `include/neml2/tensors/`) — different patterns and different test binary (`tensor_tests`).
- Adding a `Driver`, `Solver`, or `WorkScheduler` — different base class, different test setup.
- Creating a brand-new top-level submodule library (a sibling of `models/`, `solvers/`, `drivers/`, etc.). That's a rare event that needs a `neml2_add_submodule(...)` call, a `link_anchor.cxx`, and a force-link symbol declared in `include/neml2/neml2.h`. Surface that to the user instead of attempting it.

Adding a brand-new *domain directory* under an existing submodule (e.g. `models/electrochemistry/`) is fine — `models/` is already a submodule and its CMake glob is recursive, so any new subdirectory is picked up automatically.

## Required information from the user

Before writing files, confirm:

1. **Class name** in PascalCase (e.g., `MyVoceHardening`).
2. **Domain** — must be one of the existing directories under `include/neml2/models/`. Verify with `ls include/neml2/models/` if unsure. Common choices: `solid_mechanics`, `solid_mechanics/elasticity`, `solid_mechanics/crystal_plasticity`, `crystallography`, `chemical_reactions`, `phase_field_fracture`, `porous_flow`, `finite_volume`, `common`. The directory may be one or two levels deep.
3. **Base class** — default `Model`. Many solid-mechanics classes subclass a domain interface instead so that consumers can swap implementations: yield functions subclass `YieldFunction`, isotropic-hardening laws subclass `IsotropicHardening`, flow rules subclass `FlowRule`, etc. If the user describes the model in those terms, prefer the domain base. When unsure, grep `class .* : public ` in the target header directory and follow the dominant pattern.
4. **Input/output variables** — what tensor types come in and go out. Common types: `Scalar`, `Vec`, `Rot`, `R2`, `SR2`, `WR2`, `R3`, `R4`, `SSR4`, `WSR4`. Users often specify these implicitly ("maps strain to stress" → input `SR2`, output `SR2`).

Use the variable names the user gives, or the names that match the domain convention (e.g., `stress`/`strain` in solid mechanics) — input and output names appear in user-facing input files, so guessing leads to renames later. Ask if you can't tell what the convention should be.

## Files to generate

For a class `<Name>` in domain `<Domain>` (which may contain `/`, e.g. `solid_mechanics/elasticity`), generate three files using the templates below. The templates cover the common shape: `: public Model`, one input, one output, no parameters.

For anything more specialized — multiple inputs/outputs of different tensor kinds, parameterized models, models that subclass a domain interface like `YieldFunction` or `Elasticity`, or domains with their own conventions like `crystal_plasticity` — read one nearby existing model in the same directory first and adapt its pattern. The references at the bottom of this skill list good starting points.

### License header (both `.h` and `.cxx`)

Every C++ file must begin with the project's standard MIT copyright header — `scripts/check_copyright.py` does an exact textual check, so paraphrasing fails CI. The header is the contents of the repo-root `LICENSE` file with each line prefixed by `// ` (and `//` for the blank lines between paragraphs). Read `LICENSE` and reproduce it that way. Match the year in `LICENSE` exactly; don't bump it for new files.

If unsure of the format, mirror the top of `src/neml2/models/common/RotationMatrix.cxx` (or any other `.cxx`/`.h` in the tree).

### 1. Header — `include/neml2/models/<Domain>/<Name>.h`

```cpp
// <project MIT header — see "License header" section above>

#pragma once

#include "neml2/models/Model.h"   // or e.g. ".../solid_mechanics/IsotropicHardening.h" for a non-Model base

namespace neml2
{
class <InputType>;   // forward declarations of tensor types used only by reference/pointer
class <OutputType>;

/**
 * @brief <One-line summary the user gives. The first sentence becomes the syntax-doc summary.>
 */
class <Name> : public <Base>
{
public:
  static OptionSet expected_options();

  <Name>(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  const Variable<<InputType>> & _<input_name>;

  Variable<<OutputType>> & _<output_name>;
};
} // namespace neml2
```

### 2. Source — `src/neml2/models/<Domain>/<Name>.cxx`

```cpp
// <project MIT header — see "License header" section above>

#include "neml2/models/<Domain>/<Name>.h"
#include "neml2/tensors/<InputType>.h"
#include "neml2/tensors/<OutputType>.h"

namespace neml2
{
register_NEML2_object(<Name>);

OptionSet
<Name>::expected_options()
{
  OptionSet options = <Base>::expected_options();
  options.doc() = "<Same one-line summary used in the header docstring.>";

  options.add_input("<input_name>", "<Short description of the input.>");
  options.add_output("<output_name>", "<Short description of the output.>");

  return options;
}

<Name>::<Name>(const OptionSet & options)
  : <Base>(options),
    _<input_name>(declare_input_variable<<InputType>>("<input_name>")),
    _<output_name>(declare_output_variable<<OutputType>>("<output_name>"))
{
}

void
<Name>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // TODO: implement the forward operator.
  if (out)
    _<output_name> = /* function of _<input_name>() */;

  // TODO: implement the first derivative. Until this is filled in,
  // the unit test must set check_first_derivatives = false.
  if (dout_din)
    _<output_name>.d(_<input_name>) = /* d output / d input */;
}
} // namespace neml2
```

Notes on the source template:
- `register_NEML2_object(<Name>);` is mandatory. Without it the type is silently unknown to the factory at runtime.
- `_<input_name>()` (with parentheses) returns the underlying tensor of the input variable; `_<output_name> = …` and `_<output_name>.d(_<input_name>) = …` write the forward value and the derivative respectively.
- If the model has parameters (calibratable scalars/tensors), declare them with `declare_parameter<...>("name")` and add a matching `options.add<...>("name")` in `expected_options()`. Look at `LinearIsotropicHardening` or `FredrickArmstrongPlasticHardening` for parameterized examples.
- If `set_value` provides analytic first derivatives, NEML2 will cross-check them against finite differencing in the unit test (see step 3). If you can't yet provide them, set `check_first_derivatives = false` in the `.i` file.

### 3. Unit-test input — `tests/unit/models/<Domain>/<Name>.i`

The test binary `unit_tests` auto-discovers every `.i` file under `tests/unit/models/`. There is no need to add the file anywhere else.

```hit
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_<TENSOR_KIND>_names = '<input_name>'
    input_<TENSOR_KIND>_values = '<input_value_tensor>'
    output_<TENSOR_KIND>_names = '<output_name>'
    output_<TENSOR_KIND>_values = '<expected_output_tensor>'
    # Set to false if analytic derivatives are not yet implemented:
    check_first_derivatives = true
    derivative_abs_tol = 1e-6
    derivative_rel_tol = 1e-4
  []
[]

[Tensors]
  [<input_value_tensor>]
    type = Fill<TensorType>
    values = '<comma-or-space-separated values>'
  []
  [<expected_output_tensor>]
    type = Fill<TensorType>
    values = '<comma-or-space-separated values>'
  []
[]

[Models]
  [model]
    type = <Name>
    <input_name> = '<input_name>'
    <output_name> = '<output_name>'
    # Add any parameters the model declares here.
  []
[]
```

`<TENSOR_KIND>` is the suffix used by `ModelUnitTest`'s option groups. The common ones are `Scalar`, `SR2`, `R2`, `Rot`, `Vec`, `SSR4`. If the model has multiple inputs/outputs of different kinds, repeat the `input_<KIND>_names` / `input_<KIND>_values` pair (and likewise for outputs) for each kind.

If you don't have known reference values yet, leave `<expected_output_tensor>` filled with placeholder zeros and tell the user to substitute real numbers before committing.

## After scaffolding

Print a short next-steps message to the user:

```
Scaffolded:
  include/neml2/models/<Domain>/<Name>.h
  src/neml2/models/<Domain>/<Name>.cxx
  tests/unit/models/<Domain>/<Name>.i

Build:
  cmake --build --preset dev

Run only this test:
  ./build/dev/tests/unit/unit_tests models -c <Domain>/<Name>.i

CMake auto-discovers the new files via file(GLOB_RECURSE) — no CMakeLists.txt edits are needed.

Before committing:
  1. Fill in the set_value forward op and derivatives in <Name>.cxx.
  2. Replace the placeholder tensor values in <Name>.i with actual reference data.
  3. Keep check_first_derivatives = true once derivatives are correct (NEML2 cross-checks them against finite differencing).
  4. clang-format will rewrite the files on pre-commit; that's expected.
```

## Easy mistakes to verify against

These are the failure modes worth a second look — most of them compile fine and surface only at runtime or test time:

- **Missing `register_NEML2_object(<Name>);`** in the `.cxx`. Without it the type loads but the factory doesn't know about it; the `.i` test fails with "unknown type". The macro must be at namespace scope, outside any function.
- **Tensor types used by value but only forward-declared.** Anything used as a value (constructed, returned, arithmetic on) needs a full `#include "neml2/tensors/<T>.h"` in the `.cxx`. Forward declaration in the `.h` is fine for member types held by reference.
- **`.i` options that don't match `expected_options()`** — every input/output declared in `expected_options()` must be set in the `[Models]` block, and vice versa. The HIT parser fails fast and the message is clear, but it's easy to miss when copying templates.

## Reference: canonical example to follow

A minimal-but-complete `: public Model` implementation in the codebase to mirror is `RotationMatrix`:
- `include/neml2/models/common/RotationMatrix.h`
- `src/neml2/models/common/RotationMatrix.cxx`

For a parameterized Model with multiple inputs and a derivative w.r.t. parameters, see:
- `include/neml2/models/solid_mechanics/LinearIsotropicHardening.h`

For a Model that subclasses a domain base (not `Model` directly), see:
- `include/neml2/models/solid_mechanics/elasticity/LinearIsotropicElasticity.h`

The official documentation walkthrough is `doc/content/tutorials/contributing.md` and the [extension](doc/content/tutorials/extension/) tutorial set.
