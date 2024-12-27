# Development Environment {#dev-env}

[TOC]

## Configure and build (revisted)

The CMake configure and build commands were introduced in the [Installation Guide](@ref install-user). The commands were

```
cmake --preset release -S .
cmake --build --preset release-cpp
```
which uses the CMake preset named "release". The configure and build commands can take other CMake presets in the following form

```
cmake --preset {configurePreset} -S .
cmake --build --preset {buildPreset}
```
where `{configurePreset}` and `{buildPreset}` should be substituted with a configure preset and a build preset, respectively.
In addition to the "release" preset, NEML2 offers a few configure and build presets useful for typical development workflows, defined in `CMakePresets.json`. Typical workflows and their corresponding presets are listed below.

### Developing the C++ backend

```
cmake --preset dev -S .
cmake --build --preset dev-cpp
```
See also [Testing/C++ backend/Setup](@ref testing-cpp) for how to run tests once the library is built.

Unlike the "release" preset, the "dev" preset builds NEML2 with debug info (i.e., `-g`). Debuggers (gdb, lldb, etc.) can be used to step through the execution, set break points, navigate through backtrace, etc.

### Developing the Python package

```
cmake --preset dev -S .
cmake --build --preset dev-python
```
See also [Testing/Python package/Setup](@ref testing-python) for how to run tests once the package is built.

### Writing documentation

```
cmake --preset dev -S .
cmake --build --preset dev-doc
```
Once the documentation is built, the site can be previewed locally using
```
firefox build/dev/doc/build/html/index.html
```
Feel free to replace `firefox` with other browsers that support rendering of static html pages, e.g., `google-chrome`.

### Benchmarking

```
cmake --preset benchmark -S .
cmake --build benchmark
```
The "benchmark" preset builds the core C++ library along with the tests. In addition, it builds a simple [runner](@ref runner) to parse and run arbitrary input files. All test executables and the runner are linked against gperftools' CPU profiler for profiling purposes.

### Coverage

```
cmake --preset coverage -S .
cmake --build coverage
```
The unit testing executable is built with coverage flags set. Standard code coverage tools such as `gcov` and `lcov` can be used to capture and record coverage data.

### Overriding presets

While the default presets should cover most of the development workflows, it is sometimes necessary to override certain options at various stages. In general, there are three ways of overriding the preset:
- Command line options
- Environment variables
- [CMakeUserPresets.json](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)

All three options are fully supported by the cmake-tools vscode extension.

## Code formatting and linting

The C++ source code is formatted using `clang-format`. A `.clang-format` file is provided at the repository root specifying the formatting requirements. When using an IDE providing plugins or extensions to formatting C++ source code, it is important to
1. Point the plugin/extension to use the `.clang-format` file located at NEML2's repository root.
2. Associate file extensions `.h` and `.cxx` with C++.

The Python scripts shall be formatted using `black`. Formatting requirements are specified under the `[black]` section in `pyproject.toml`.

All pull requests will be run through `clang-format` and `black` to ensure formatting consistency.

For linting, a `.clang-tidy` file is provided at the repository root to specify expected checks.
