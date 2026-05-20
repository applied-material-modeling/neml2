# Basic Installation {#install}

[TOC]

## Prerequisites

\note
The prerequisites below apply to **source builds** (C++ library or Python package built from source). If you are installing the Python package via the pre-built wheels on PyPI, you only need Python and PyTorch — see the [Python package](#install-python) section below.

Compiling the NEML2 core library requires
- A C++ compiler with C++17 support
<!-- dependencies: python.version_min -->
- [Python](https://www.python.org/downloads/) >= 3.10
<!-- dependencies: torch.version -->
- [PyTorch](https://pytorch.org/get-started/locally/) ~= 2.12.0
<!-- dependencies: cmake.version_min -->
- [CMake](https://cmake.org/download/) >= 3.26.1

NEML2 is built upon and relies on a collection of third party libraries/packages. The build system will automatically search for necessary packages and download missing dependencies when appropriate.

Refer to [dependency management](@ref dependency-management) for a list of dependent libraries/packages and more information on the search procedure.

## Build {#install-user}

\note
NEML2 is available both as a C++ library and as a Python package. Instructions for each variant are provided below. If you only need one of them, the other can be skipped.

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

### Python package {#install-python}

NEML2 offers a Python package with bindings for the primitive tensors and parsers for deserializing and running material models.

#### Installing from PyPI (recommended)

Pre-built wheels are available on [PyPI](https://pypi.org/project/neml2/) for Linux and macOS:

```shell
pip install neml2
```

[PyTorch](https://pytorch.org/get-started/locally/) is listed as a dependency and will be installed automatically if not already present. Each NEML2 wheel is verified against a fixed set of torch versions — see [torch compatibility](@ref torch-compat) for the current list. Wheels ship with Python stub files pre-generated, so IDE autocomplete and type hints work out of the box — no extra steps are needed.

Once installed, the package can be imported in Python scripts using

```python
import neml2
```

#### Building from source

To build and install the Python package from source (e.g., to modify the C++ source or build against a custom LibTorch), first make sure Python development headers and libraries are installed and [PyTorch](https://pytorch.org/get-started/locally/) is importable. Then, at the repository root:

```shell
pip install .
```

The installation will take a while as it compiles both the library and its bindings. Pass `-v` to see additional details. After installation from source, generate Python stub files with
```shell
neml2-stub
```
These stubs improve IDE autocomplete, type hints, and documentation generation for compiled bindings.
