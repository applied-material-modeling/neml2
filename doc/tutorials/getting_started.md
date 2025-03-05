@insert-title:tutorials-getting-started

[TOC]

Once the NEML2 library is built following the [installation guide](@ref install), the immediate question is

> How do I evaluate a NEML2 material model?

## Downstream C++ project

NEML2 core capabilities are implemented as a _library_, not a _program_. As a library, it shall be used in another C++ program to parse and evaluate a material model defined in an [input file](@ref tutorials-models-input-file). Boilerplate for the C++ program can be found in this [set of tutorials](#tutorials-models), and external project integration is documented in the [installation guide](@ref external-project-integration).

## The Runner

We acknowledge the common need to use NEML2 as a standalone program and therefore provide two convenient options for users to effectively use NEML2 as a standalone program, with the first option being the *NEML2 Runner*.

As documented in [build customization](@ref build-customization), the "runner" preset or the `NEML2_RUNNER` CMake option can be used to create a simple "runner" program for parsing, diagnosing, and running NEML2 material models.
```
cmake --preset runner -S .
cmake --build --preset runner
```

Once the runner is built (and/or installed), an executable named `runner` will be placed under the build directory (or the installation directory). Invoking the executable without any additional argument or with the `-h` or `--help` argument will print out the usage message:
```
Usage: runner [--help] [--version] [--diagnose] [--time] [--num-runs VAR] [--warmup VAR] input driver additional_args

Positional arguments:
  input            path to the input file
  driver           name of the driver in the input file
  additional_args  additional command-line arguments to pass to the input file parser [nargs: 0 or more]

Optional arguments:
  -h, --help       shows help message and exits
  -v, --version    prints version information and exits
  -d, --diagnose   run diagnostics on common problems and exit if any issue is identified
  -t, --time       output the elapsed wall time during model evaluation
  -n, --num-runs   number of times to run the driver [nargs=0..1] [default: 1]
  --warmup         number of warmup runs before actually measuring the model evaluation time [nargs=0..1] [default: 0]
```

## Python script

The other option to evaluate NEML2 material models is to use the NEML2 Pyton package. As mentioned in the [installation guide](@ref install), NEML2 also provides an _experimental_ Python package which provides bindings for the primitive tensors and parsers for deserializing and running material models. This [set of tutorials](#tutorials-models) describe the usage of the package.
