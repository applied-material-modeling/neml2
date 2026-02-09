# Basic Installation {#install}

[TOC]

## Prerequisites

Compiling the NEML2 core library requires
- A C++ compiler with C++17 support
- CMake >= 3.26
- Python >= 3.9, or its libtorch distribution

NEML2 is built upon and relies on a collection of third party libraries/packages, most notably PyTorch (or its C++ backend libtorch). The build system will automatically search for necessary packages and download missing dependencies when appropriate.

Refer to [dependency management](@ref dependency-management) for a list of dependent libraries/packages and more information on the search procedure.

## Build {#install-user}

\note
NEML2 is available both as a C++ library and as a Python package. Instructions for building and installing each variant are provided below. If you only need one of them, the other can be skipped.

### C++ backend

First, obtain the NEML2 source code.

```shell
git clone -b main https://github.com/applied-material-modeling/neml2.git
```

Then, configure and build NEML2.

```shell
cmake --preset release -S .
cmake --build --preset release
```
The `--preset` option specifies a predefined configuration to be read by CMake. Build customization, available presets, and their usage scenario are discussed in [build customization](@ref build-customization).

Optionally, NEML2 can be installed as a system library.
```shell
cmake --install build/release --component libneml2 --prefix /usr/local
```
The `--prefix` option specifies the path where NEML2 will be installed. Write permission is needed for the installation path. The `--component libneml2` option tells CMake to only install the libraries and runtime artifacts. NEML2 has two installable components:
- `libneml2`: core libraries and public headers
- `libneml2-bin`: utility binaries

### Python package

NEML2 also offers a Python package which provides bindings for the primitive tensors and parsers for deserializing and running material models. Package source distributions are available on PyPI. Python wheel deployment is planned and currently under development.

To install the NEML2 Python package, first make sure Python development headers and libraries are installed. Then, install [PyTorch](https://pytorch.org/get-started/locally/) and run the following command at the repository's root.

```shell
pip install .
```

The installation will take a while, as it needs to build both the library and its bindings. Pass `-v` to see additional details.
After installation from source, optionally generate Python stub files with
```shell
neml2-stub
```
These stubs improve IDE autocomplete, type hints, and documentation generation for compiled bindings.

\note
Planned PyPI wheels are expected to ship with stubs pre-generated, in which case the `neml2-stub` step will no longer be needed for wheel installs.

Once installed, the package can be imported in Python scripts using

```python
import neml2
```
