@insert-title:tutorials-getting-started

[TOC]

Once the NEML2 library is built following the [installation guide](@ref install), the immediate question is

> How do I evaluate a NEML2 material model?

## Downstream C++ project

NEML2 core capabilities are implemented as a _library_, not a _program_. As a library, it will be used by another C++ program to parse and evaluate a material model defined in an [input file](@ref tutorials-models-input-file). Boilerplate for the C++ program can be found in this [set of tutorials](#tutorials-models), and external project integration is documented in the [installation guide](@ref external-project-integration).

## Utility binaries

We acknowledge the common need to use NEML2 as a standalone program and therefore provide utility binaries for common tasks.

As documented in [build customization](@ref build-customization), the `NEML2_TOOLS` CMake option can be used to create these binaries.

When tools are enabled, NEML2 builds the following standalone executables:
- `diagnose`
- `inspect`
- `run`
- `syntax`
- `time`

When installed, these binaries are placed in the installed `bin` directory.

For the Python package, matching CLI wrappers are registered in `pyproject.toml`:
- `neml2-diagnose`
- `neml2-inspect`
- `neml2-run`
- `neml2-syntax`
- `neml2-time`

These commands dispatch to the shipped binaries bundled inside the Python package. See each tool's help message (for example, `neml2-run --help`) for further details.

## Python script

The other option to evaluate NEML2 material models is to use the NEML2 Python package. As mentioned in the [installation guide](@ref install), NEML2 also provides a Python package which provides bindings for the primitive tensors and parsers for deserializing and running material models. This [set of tutorials](#tutorials-models) describes the usage of the package.
