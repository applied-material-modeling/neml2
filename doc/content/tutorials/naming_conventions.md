@insert-title:tutorials-naming-conventions

[TOC]

## Naming conventions

### Variable naming conventions

The following characters are not allowed in variable names:
- whitespace characters: input file parsing ambiguity
- `,`: input file parsing ambiguity
- `;`: input file parsing ambiguity
- `.`: clash with PyTorch parameter/buffer naming convention

### Source code naming conventions

In NEML2 source code, the following naming conventions are recommended:
- User-facing variables and option names should be _as descriptive as possible_. For example, the equivalent plastic strain is named "equivalent_plastic_strain". Note that white spaces, quotes, and left slashes are not allowed in the names. Underscores are recommended as a replacement for white spaces.
- Developer-facing variables and option names should use simple alphanumeric symbols. For example, the equivalent plastic strain is named "ep" in consistency with most of the existing literature.
- Developer-facing member variables and option names should use the same alphanumeric symbols. For example, the member variable for the equivalent plastic strain is named `ep`. However, if the member variable is protected or private, it is recommended to prefix it with an underscore, i.e. `_ep`.
- Struct names and class names should use `PascalCase`.
- Function names should use `snake_case`.
