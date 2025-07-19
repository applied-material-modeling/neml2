@insert-title:tutorials-naming-conventions

[TOC]

## Naming conventions

### Reserved axis names

NEML2 models group variables onto different axes by their names, and the collection of variable names (with their corresponding layout) is called a labeled axis ([LabeledAxis](@ref neml2::LabeledAxis)). NEML2 predefines six sub-axes to categorize all the input, output and intermediate variables:
- State \f$\mathcal{S}\f$ (axis name `state`): Variables collectively characterizing the current _state_ of the material subject to given external forces. The state variables are usually the output of a physically meaningful material model.
- Forces \f$\mathcal{F}\f$ (axis name `forces`): Variables defining the _external_ forces that drive the response of the material.
- Old state \f$\mathcal{S}_n\f$ (axis name `old_state`): The state variables _prior to_ the current material update. In the time-discrete setting, these are the state variables from the previous time step.
- Old forces \f$\mathcal{F}_n\f$ (axis name `old_forces`): The external forces _prior to_ the current material update. In the time-discrete setting, these are the forces from the previous time step.
- Residual \f$\mathcal{R}\f$ (axis name `residual`): The residual defines an _implicit_ model/function. An implicit model is updated by solving for the state variables that result in zero residual.
- Parameters \f$\mathcal{P}\f$ (axis name `parameters`): The (nonlinear) parameters.

\note
When authoring C++ source code, it is recommended to avoid hard-coding reserved axis names as pure strings. Instead, inlined `const` string names (defined in `neml2/models/LabeledAxisAccessor.h`) should be used wherever possible, i.e. `STATE`, `OLD_STATE`, `FORCES`, `OLD_FORCES`, `RESIDUAL`, and `PARAMETERS`.

### Variable naming conventions

Variable names have the type neml2::VariableName which is an alias to neml2::LabeledAxisAccessor. The following characters are not allowed in variable names:
- whitespace characters: input file parsing ambiguity
- `,`: input file parsing ambiguity
- `;`: input file parsing ambiguity
- `.`: clash with PyTorch parameter/buffer naming convention
- `/`: separator reserved for nested variable name

In the input file, the separator `/` is used to denote nested variable names. For example, `A/B/foo` specifies a variable named "foo" defined on the sub-axis named "B" which is a nested sub-axis of "A".

### Source code naming conventions

In NEML2 source code, the following naming conventions are recommended:
- User-facing variables and option names should be _as descriptive as possible_. For example, the equivalent plastic strain is named "equivalent_plastic_strain". Note that white spaces, quotes, and left slashes are not allowed in the names. Underscores are recommended as a replacement for white spaces.
- Developer-facing variables and option names should use simple alphanumeric symbols. For example, the equivalent plastic strain is named "ep" in consistency with most of the existing literature.
- Developer-facing member variables and option names should use the same alphanumeric symbols. For example, the member variable for the equivalent plastic strain is named `ep`. However, if the member variable is protected or private, it is recommended to prefix it with an underscore, i.e. `_ep`.
- Struct names and class names should use `PascalCase`.
- Function names should use `snake_case`.
