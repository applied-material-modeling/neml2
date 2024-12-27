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

## Build and install {#install-user}

\note
NEML2 is available both as a C++ library and as a Python package. Instructions for building and installing each variant are provided below. If you only need one of them, the other can be skipped.

### C++ backend

First, obtain the NEML2 source code.

```
git clone https://github.com/applied-material-modeling/neml2.git
cd neml2
git checkout main
```

Then, configure NEML2.

```
cmake --preset release -S .
```

Finally, compile NEML2.

```
cmake --build --preset release-cpp
```

\note
If the generator was configured to be "Unix Makefiles" (the default for Unix-based operating systems), an additional `-j N` option could be passed through the commandline to accelerate the build process by running `N` jobs in parallel.

The compiled NEML2 can be installed as a system library.

```
cmake --install build/release --prefix /usr/local
```

\note
The `--prefix` option specifies the path where NEML2 will be installed. Write permission is needed for the installation path.

For more fine-grained control over the configure, build, and install commands, please refer to the [CMake User Interaction Guide](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html).


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

CMake presets are provided by NEML2 to simplify the installation process. The presets are defined in `CMakePresets.json`. While presets are useful in getting a quick start, finer control is oftentimes needed during installation. This section documents how the build can be customized without the presets.

Commonly used configuration options are summarized below. Default options are <u>underlined</u>.

| Option             | Values (<u>default</u>) | Description                                                    |
| :----------------- | :---------------------- | :------------------------------------------------------------- |
| NEML2_TESTS        | <u>ON</u>, OFF          | Master knob for including/excluding all tests                  |
| NEML2_RUNNER       | ON, <u>OFF</u>          | Create a simple runner                                         |
| NEML2_CPU_PROFILER | ON, <u>OFF</u>          | Linking against gperftools libprofiler to enable CPU profiling |
| NEML2_DOC          | ON, <u>OFF</u>          | Create the documentation target                                |
| NEML2_PYBIND       | ON, <u>OFF</u>          | Create the Python bindings target                              |

Additional configuration options can be passed via command line using the `-DOPTION` or `-DOPTION=ON` format. For example,

```
cmake -DNEML2_PYBIND=ON -B build .
```
turns on the `NEML2_PYBIND` option, and additional targets for building the Python bindings will be created. Note that this would also download additional optional dependencies, e.g., pybind11, that are required to build the Python bindings.

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


