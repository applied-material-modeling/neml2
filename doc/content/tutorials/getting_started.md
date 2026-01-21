@insert-title:tutorials-getting-started

[TOC]

Once the NEML2 library is built following the [installation guide](@ref install), the immediate question is

> How do I evaluate a NEML2 material model?

## Downstream C++ project

NEML2 core capabilities are implemented as a _library_, not a _program_. As a library, it will be used by another C++ program to parse and evaluate a material model defined in an [input file](@ref tutorials-models-input-file). Boilerplate for the C++ program can be found in this [set of tutorials](#tutorials-models), and external project integration is documented in the [installation guide](@ref external-project-integration).

## The runner

We acknowledge the common need to use NEML2 as a standalone program and therefore provide two convenient options for users to effectively use NEML2 as a standalone program, with the first option being the *NEML2 runner*.

As documented in [build customization](@ref build-customization), the "release" preset or the `NEML2_RUNNER` CMake option can be used to create a convenient executable for parsing, diagnosing, and running NEML2 material models.
```
cmake --preset release -S .
cmake --build --preset release
```

Once the runner is built (and/or installed), an executable named `neml2` will be placed under the build directory (or the installation directory). Invoking the executable without any additional argument or with the `-h` or `--help` argument will print out the usage message:
```
Usage: neml2 [--help] [--version] {diagnose,inspect,run,time}

Optional arguments:
  -h, --help     shows help message and exits
  -v, --version  prints version information and exits

Subcommands:
  diagnose      Run diagnostics on a driver or a model from an input file.
  inspect       Summarize the structure of a model.
  run           Run a driver from an input file.
  time          Time the execution of a driver from an input file.
```
See the subcommand help messages for further details.

## Python script

The other option to evaluate NEML2 material models is to use the NEML2 Python package. As mentioned in the [installation guide](@ref install), NEML2 also provides an _experimental_ Python package which provides bindings for the primitive tensors and parsers for deserializing and running material models. This [set of tutorials](#tutorials-models) describes the usage of the package.
