# Frequently Asked Questions {#faq}

## How do I run a NEML2 input file? {#runner}

NEML2 is designed as a _library_, not a _program_: The core library itself cannot be used as a standalone program to parse and evaluate a material model defined in an input file. However, we acknowledge this common need and therefore provide two convenient options for users to effectively use NEML2 as a standalone program:
1. The NEML2 Runner: As documented in [build customization](@ref dev-env-build), the `NEML2_RUNNER` CMake option can be turned on to create a simple runner for parsing, diagnosing, running and profiling NEML2 material models. Once the runner is built, an executable will be placed inside the `runner` directory under the build directory. Invoking the executable without any additional argument or with the `-h` or `--help` argument will print out the usage message:
```
Usage: runner [--help] [--version] [--diagnose] [--time] [--num-runs VAR] [--warmup VAR] input driver additional_args

Positional arguments:
  input            path to the input file
  driver           name of the driver in the input file
  additional_args  additional command-line arguments to pass to the input file parser [nargs: 0 or more]

Optional arguments:
  -h, --help       shows help message and exits
  -v, --version    prints version information and exits
  -d, --diagnose   run diagnostics on common problems and exit (without further execution)
  -t, --time       output the elapsed wall time during model evaluation
  -n, --num-runs   number of times to run the driver [nargs=0..1] [default: 1]
  --warmup         number of warmup runs before actually measuring the model evaluation time [nargs=0..1] [default: 0]
```
2. The NEML2 Pyton package: As mentioned in the [installation guide](@ref install), NEML2 also provides an _experimental_ Python package which provides bindings for the primitive tensors and parsers for deserializing and running material models. Once the NEML2 Python package is successfully installed, you can follow the [user guide for NEML2 Python package](@ref python-package) to evaluate material models with given inputs.

In addition to the above options, you can always write a simple C++ program and link against NEML2. Boilerplate for the C++ program can be found in the [user guide](@ref cpp-backend), and CMake integration is documented in the [installation guide](@ref cmake-integration).
