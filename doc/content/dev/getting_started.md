# Gettomg Started {#dev-getting-started}

[TOC]

## Model development

Although NEML2 comes with a large collection modular building blocks for composing material models, it is sometimes necessary to write your own material models (and integrate them with existing NEML2 models). The [custom model](@ref custom-model) development guide demonstrates how a custom model can be implemented within the NEML2 framework and provides in-depth explanation for every step throughout the tutorial.

## Testing

See [Testing/C++ backend/Setup](@ref testing-cpp) for how to run tests for the C++ backend.

See [Testing/Python package/Setup](@ref testing-python) for how to run tests for the Python package.

## Documentation

It is of paramount importance to write documentation as the library is being developed. While NEML2 supports both Doxygen-style in-code documentation mechanisms and [run-time syntax documentation mechanisms](@ref custom-model-in-code-documentation), it is still sometimes necessary to write standalone, self-contained documentation.

To this end, the "dev" configure preset and the "dev-doc" build preset (see [build customization](@ref build-customization)) can be used to generate and render the documentation locally:
```
cmake --preset dev -S .
cmake --build --preset dev-doc
```
Once the documentation is built, the site can be previewed locally in any browser that supports static HTML, i.e.
```
firefox build/dev/doc/build/html/index.html
```

## Code formatting and linting

The C++ source code is formatted using `clang-format`. A `.clang-format` file is provided at the repository root specifying the formatting requirements. When using an IDE providing plugins or extensions to format C++ source code, it is important to
1. Point the plugin/extension to use the `.clang-format` file located at NEML2's repository root.
2. Associate file extensions `.h` and `.cxx` with C++.

The Python scripts shall be formatted using `black`. Formatting requirements are specified under the `[black]` section in `pyproject.toml`.

All pull requests will be run through `clang-format` and `black` to ensure formatting consistency.

For C++ linting, a `.clang-tidy` file is provided at the repository root to specify expected checks. Python linting is not currently enforced.
