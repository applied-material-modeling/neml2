# Installation Guide {#install}

[TOC]

## Prerequisites

Compiling the NEML2 core library requires
- A C++ compiler with C++17 support
- CMake >= 3.28

NEML2 is built upon and relies on a collection of third party libraries/packages, most notably PyTorch (and its C++ backend libtorch). However, **there is no need to install dependencies prior to installing NEML2**. The build system will automatically search for necessary packages. When a required package is missing, the build system will download one.

Refer to [dependency management](@ref dependency-management) for a list of dependent libraries/packages and more information on the search procedure.

## Build {#install-user}

\note
NEML2 is available both as a C++ library and as a Python package. Instructions for building and installing each variant are provided below. If you only need one of them, the other can be skipped.

### C++ backend

First, obtain the NEML2 source code.

```
git clone -b main https://github.com/applied-material-modeling/neml2.git
```

Then, configure and build NEML2.

```
cd neml2
cmake --preset release -S .
cmake --build --preset release
```
The `--preset` option specifies predefined configuration used by CMake. Available presets and their usage scenario are discussed in the [development guide](@ref dev-env).

Optionally, NEML2 can be installed as a system library.

```
cmake --install build/release --component libneml2 --prefix /usr/local
```
The `--prefix` option specifies the path where NEML2 will be installed. Write permission is needed for the installation path. The `--component libneml2` option tells CMake to only install the libraries and runtime artifacts. Refer to [components] for all installable components.

Refer to the [cmake manual](https://cmake.org/cmake/help/latest/manual/cmake.1.html) for more command line options. For more fine-grained control over the configure, build, and install commands, please refer to the [CMake User Interaction Guide](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html).


### Python package

NEML2 also provides an _experimental_ Python package which provides bindings for the primitive tensors and parsers for deserializing and running material models. Package source distributions are available on PyPI, but package wheels are currently not built and uploaded to PyPI.

To install the NEML2 Python package, run the following command at the repository's root.

```
pip install -v .
```
Two optional environment variables can be used to control the build: `CMAKE_GENERATOR` specifies the CMake generator, and `CMAKE_BUILD_JOBS` specifies the number of parallel jobs used to build the package.

Once installed, the package can be imported in Python scripts using

```python
import neml2
```

For security reasons, static analysis tools and IDEs for Python usually refuse to extract function signature, type hints, etc. from binary extensions such as the NEML2 Python bindings. As a workaround, NEML2 automatically generates "stubs" using `pybind11-stubgen` immediately after Python bindings are built to make them less opaque. Refer to the [pybind11-stubgen documentation](https://pypi.org/project/pybind11-stubgen/) for more information.
