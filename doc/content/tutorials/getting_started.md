@insert-title:tutorials-getting-started

[TOC]

Once the NEML2 library is built following the [installation guide](@ref install), the immediate question is

> How do I evaluate a NEML2 material model?

## Downstream C++ project

NEML2 core capabilities are implemented as a _library_, not a _program_. As a library, it will be used by another C++ program to parse and evaluate a material model defined in an [input file](@ref tutorials-models-input-file). Boilerplate for the C++ program can be found in the [quick start guide](#getting-started-quick-start) below as well as in this [set of tutorials](#tutorials-models), and external project integration is documented in the [installation guide](@ref external-project-integration).

## Python script

The other option to evaluate NEML2 material models is to use the NEML2 Python package. As mentioned in the [installation guide](@ref install), NEML2 also provides a Python package which provides bindings for the primitive tensors and parsers for deserializing and running material models. The following [quick start guide](#getting-started-quick-start) as well as this [set of tutorials](#tutorials-models) describe the usage of the package.

## Quick start {#getting-started-quick-start}

The following is a quick summary of the most common model API usage patterns. Each topic is covered in full detail in the [model tutorials](#tutorials-models).

### Loading a model

Given an [input file](@ref tutorials-models-input-file), a model is loaded with a single call:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:running_your_first_model/ex1.cxx

- <b class="tab-title">Python</b>
  @list:python:running_your_first_model/ex2.py

</div>

`load_model` parses the file and returns the named model ready for evaluation. The [running your first model tutorial](#tutorials-models-running-your-first-model) walks through a complete example.

### Calling forward operators

All NEML2 models expose three forward operators that accept a map of input variable values:

| Operator           | Returns                                           |
| ------------------ | ------------------------------------------------- |
| `value`            | output variable values \f$ y = f(x) \f$           |
| `dvalue`           | first derivatives \f$ \partial y / \partial x \f$ |
| `value_and_dvalue` | both simultaneously                               |

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:running_your_first_model/ex3.cxx

- <b class="tab-title">Python</b>
  @list:python:running_your_first_model/ex4.py

</div>

The return value of neml2::Model::value is a dictionary keyed by output variable name. neml2::Model::dvalue returns a nested dictionary keyed by `(output, input)` variable names.

### Modifying parameter values

Model parameters can be updated at runtime — useful for optimization or training loops. Parameters are accessed by name and assigned new tensor values:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:model_parameters/ex5.cxx

- <b class="tab-title">Python</b>
  @list:python:model_parameters/ex6.py

</div>

The [model parameters tutorial](#tutorials-models-model-parameters) covers parameters and buffers in detail, including how parameters can themselves be defined by other models.

## Utility binaries

We acknowledge the common need to use NEML2 as a standalone program and therefore provide utility binaries for common tasks.

As documented in [build customization](@ref build-customization), the `NEML2_TOOLS` CMake option can be used to create these binaries.

When tools are enabled, NEML2 builds the following standalone executables:
- `neml2-diagnose`
- `neml2-inspect`
- `neml2-run`
- `neml2-syntax`
- `neml2-time`

When installed, these binaries are placed in the installed `bin` directory.

For the Python package, matching CLI wrappers are registered in `pyproject.toml`:
- `neml2-diagnose`
- `neml2-inspect`
- `neml2-run`
- `neml2-syntax`
- `neml2-time`

These commands dispatch to the shipped binaries bundled inside the Python package. See each tool's help message (for example, `neml2-run --help`) for further details.
