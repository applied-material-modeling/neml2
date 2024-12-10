# Installation Guide {#install}

[TOC]

## Prerequisites

Compiling the NEML2 core library requires
- A C++ compiler with C++17 support
- CMake >= 3.28

## Dependencies

### Required dependencies

- [PyTorch](https://pytorch.org/get-started/locally/), version 2.5.1.

Recent PyTorch releases within a few minor versions are likely to be compatible. In the PyTorch official download page, several download options are provided: conda, pip, libTorch, and source distribution.
- **Recommended**: If you choose to download PyTorch using conda or pip, the NEML2 CMake script can automatically detect and use the PyTorch installation.
- If you choose to download libTorch or build PyTorch from source, you will need to set `LIBTORCH_DIR` to be the location of libTorch when using CMake to configure NEML2.

\note
The libTorch distributions from the official website come with two flavors: "Pre-cxx11 ABI" and "cxx11 ABI". Both variants are supported by NEML2. If you are unsure, we recommend the one with "cxx11 ABI".

If no PyTorch installation can be detected and `LIBTORCH_DIR` is not set at configure time, the NEML2 CMake script will automatically download and use the libTorch obtained from the official website. Note, however, that this method only works on Linux and Mac systems.

\note
We strive to keep up with the rapid development of PyTorch. The NEML2 PyTorch dependency is updated on a quarterly basis. If there is a particular version of PyTorch you'd like to use which is found to be incompatible with NEML2, please feel free to [create an issue](https://github.com/applied-material-modeling/neml2/issues).

### Optional dependencies

\note
No action is needed to manually obtain the optional dependencies. The compatible optional dependencies will be automatically downloaded and configured by CMake depending on the build customization.

- [HIT](https://github.com/idaholab/moose/tree/master/framework/contrib/hit) for input file parsing.
- [WASP](https://code.ornl.gov/neams-workbench/wasp) as the lexing and parsing backend for HIT.
- [Catch2](https://github.com/catchorg/Catch2) for unit and regression testing.
- [gperftools](https://github.com/gperftools/gperftools) for profiling.
- [Doxygen](https://github.com/doxygen/doxygen) for building the documentation.
- [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css) the documentation theme.
- [argparse](https://github.com/p-ranav/argparse) for command-line argument parsing.
- [pybind11](https://github.com/pybind/pybind11) for building Python bindings.
- Python packages
  - [graphviz](https://github.com/xflr6/graphviz) for model visualization
  - [pytest](https://docs.pytest.org/en/stable/index.html) for testing Pythin bindings
  - [PyYAML](https://pyyaml.org/) for extracting syntax documentation
  - [pybind11-stubgen](https://github.com/sizmailov/pybind11-stubgen) for extracting stubs from Python bindings

## Build and install

\note
NEML2 is available both as a C++ library and as a Python package. Instructions for building and installing each variant are provided below. If you only need one of them, the other can be skipped.

### C++ backend

First, obtain the NEML2 source code.

```
git clone https://github.com/applied-material-modeling/neml2.git
cd neml2
git checkout main
```

Then, configure NEML2. See [build customization](#install-build-customization) for possible configuration options.

```
cmake -B build .
```

Finally, compile NEML2.

```
cmake --build build -j N
```
where `N` is the number of cores to use for parallel compilation.

The compiled NEML2 can be installed as a system library.

```
cmake --install build
```

For more fine-grained control over the configure, build, and install commands, please refer to the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html).


### Python package

NEML2 also provides an _experimental_ Python package which provides bindings for the primitive tensors and parsers for deserializing and running material models. Package source distributions are available on PyPI, but package wheels are currently not built and uploaded to PyPI.

To install the NEML2 Python package, run the following command at the repository's root. Note that unlike the C++ backend, we do not expose any interface for customizing the build. The default configuration is already optimized for building the Python package.

```
pip install -v .
```

The command installs a package named `%neml2` to the site-packages directory, and so it can be imported in Python scripts using

```python
import neml2
```

For security reasons, static analysis tools and IDEs for Python usually refuse to extract function signature, type hints, etc. from binary extensions such as the NEML2 Python bindings. As a workaround, NEML2 automatically generates "stubs" using `pybind11-stubgen` immediately after Python bindings are built to make them less opaque. Refer to the [pybind11-stubgen documentation](https://pypi.org/project/pybind11-stubgen/) for more information.

## Build customization {#install-build-customization}

Additional configuration options can be passed via command line using the `-DOPTION` or `-DOPTION=ON` format. For example,

```
cmake -DNEML2_PYBIND=ON -B build .
```
turns on the `NEML2_PYBIND` option, and additional targets for building the Python bindings will be created. Note that this would also download additional optional dependencies, e.g., pybind11, that are required to build the Python bindings.

Commonly used configuration options are summarized below. Default options are <u>underlined</u>.

| Option                | Values (<u>default</u>)                                     | Description                                                                               |
| :-------------------- | :---------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| CMAKE_BUILD_TYPE      | <u>Debug</u>, Release, MinSizeRel, RelWithDebInfo, Coverage | CMake [Reference](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)     |
| CMAKE_INSTALL_PREFIX  |                                                             | CMake [Reference](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html) |
| CMAKE_UNITY_BUILD     |                                                             | CMake [Reference](https://cmake.org/cmake/help/latest/variable/CMAKE_UNITY_BUILD.html)    |
| NEML2_TESTS           | <u>ON</u>, OFF                                              | Master knob for including/excluding all tests                                             |
| NEML2_UNIT            | <u>ON</u>, OFF                                              | Create the unit testing target                                                            |
| NEML2_REGRESSION      | <u>ON</u>, OFF                                              | Create the regression testing target                                                      |
| NEML2_VERIFICATION    | <u>ON</u>, OFF                                              | Create the verification testing target                                                    |
| NEML2_RUNNER          | ON, <u>OFF</u>                                              | Create a simple runner                                                                    |
| NEML2_CPU_PROFILER    | ON, <u>OFF</u>                                              | Linking against gperftools libprofiler to enable CPU profiling                            |
| NEML2_DOC             | ON, <u>OFF</u>                                              | Create the documentation target                                                           |
| NEML2_PYBIND          | ON, <u>OFF</u>                                              | Create the Python bindings target                                                         |
| NEML2_CLANG_TIDY      | ON, <u>OFF</u>                                              | Run clang-tidy linting diagnostics                                                        |
| NEML2_CLANG_TIDY_PATH | <u>"clang-tidy"</u>                                         | Path to clang-tidy executable                                                             |

Visual Studio Code users are encouraged to use the predefined [CMake variants](https://vector-of-bool.github.io/docs/vscode-cmake-tools/variants.html) in `cmake-variants.yaml` to configure the build.

## CMake integration {#install-cmake-integration}

Integrating NEML2 into a project that already uses CMake is fairly straightforward. The following CMakeLists.txt snippet links NEML2 into the target executable called `foo`:

```
add_subdirectory(neml2)

add_executable(foo main.cxx)
target_link_libraries(foo neml2)
```

The above snippet assumes NEML2 is checked out to the directory %neml2, i.e., as a git submodule.
Alternatively, you may use CMake's `FetchContent` module to integrate NEML2 into your project:

```
FetchContent_Declare(
  neml2
  GIT_REPOSITORY https://github.com/applied-material-modeling/neml2.git
  GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(neml2)

add_executable(foo main.cxx)
target_link_libraries(foo neml2)
```

## Testing {#install-testing}

### C++ backend

By default when `NEML2_TESTS` is set to `ON`, three test suites are built under the specified build directory:

- `tests/unit/unit_tests`: Collection of tests to ensure individual objects are working correctly.
- `tests/regression/regression_tests`: Collection of tests to avoid regression.
- `tests/verification/verification_tests`: Collection of verification problems.

The tests assume the working directory to be the `tests` directory relative to the repository root. For Visual Studio Code users, the [C++ TestMate](https://github.com/matepek/vscode-catch2-test-adapter) extension can be used to automatically discover and run tests. In the extension settings, the "Working Directory" variable should be modified to `${workspaceFolder}/tests`. The `settings.json` file shall contain the following entry:
```json
{
  "testMate.cpp.test.workingDirectory": "${workspaceFolder}/tests",
}
```

### Python package

A collection of tests are available under `python/tests` to ensure the NEML2 Python package is working correctly. For Visual Studio Code users, the [Python](https://github.com/Microsoft/vscode-python) extension can be used to automatically discover and run tests. In the extension settings, the "Pytest Enabled" variable shall be set to true. In addition, "pytestArgs" shall provide the location of tests, i.e. "${workspaceFolder}/python/tests". The `settings.json` file shall contain the following entries:
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "${workspaceFolder}/python/tests"
  ],
}
```

If the Python bindings are built (with `NEML2_PYBIND` set to `ON`) but are not installed to the site-packages directory, pytest will not be able to import the %neml2 package unless the environment variable `PYTHONPATH` is modified according to the specified build directory. For Visual Studio Code users, create a `.env` file in the repository's root and include an entry `PYTHONPATH=build/python` (assuming the build directory is `build`), and the Python extension will be able to import the NEML2 Python package.
