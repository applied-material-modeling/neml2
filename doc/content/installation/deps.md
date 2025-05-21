# Dependency management {#dependency-management}

[TOC]

## Dependency search

In most cases, there is no need to manually obtain the dependent libraries/packages. The build system will automatically search for the required packages at usual locations. If the package is already installed on the system, it will be used to build NEML2. Otherwise, a compatible version of the package will be downloaded and installed under the NEML2 build directory.

In case the package of interest has been installed at a non-conventional location, and CMake's default searching mechanism fails to find it, some special configure options can be used to help locate it. For a package named `<PackageName>`, the following variables are tried in sequence:
- `<PackageName>_ROOT` CMake variable
- `<PACKAGENAME>_ROOT` CMake variable
- `<PackageName>_ROOT` enviroment variable
- `<PACKAGENAME>_ROOT` environment variable

Please refer to the [CMake documentation](https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-search-procedure) for additional hints that can be used to facilitate the package search procedure.

The following table summarizes the names of the packages required by each configure options. The first row lists the packages required by the base library.

| Option                | Required package(s) |
| :-------------------- | :------------------ |
|                       | Torch, WASP, HIT    |
| NEML2_TESTS           | Catch2              |
| NEML2_PYBIND          | Python              |
| NEML2_DOC             | Doxygen             |
| NEML2_WORK_DISPATCHER | MPI, TIMPI          |

## List of dependencies

### C++ backend dependencies

- [PyTorch](https://pytorch.org/get-started/locally/), version 2.5.1.
- [HIT](https://github.com/idaholab/moose/tree/master/framework/contrib/hit) for input file parsing.
- [WASP](https://code.ornl.gov/neams-workbench/wasp) as the lexing and parsing backend for HIT.
- [Catch2](https://github.com/catchorg/Catch2) for unit and regression testing.

In addition to standard system library locations, the CMake configure script also searches for an installed torch Python package. Recent PyTorch releases within a few minor versions are likely to be compatible.

\warning
If no PyTorch is found after searching, a CPU-only libtorch binary is downloaded from the official website. Such libtorch is not able to use the CUDA co-processor, even if there is one. If using CUDA is desired, please install the torch Python package or a CUDA-enabled libtorch before configuring NEML2.

\note
We strive to keep up with the rapid development of PyTorch. The NEML2 PyTorch dependency is updated on a quarterly basis. If there is a particular version of PyTorch you'd like to use which is found to be incompatible with NEML2, please feel free to [create an issue](https://github.com/applied-material-modeling/neml2/issues).

### Python package dependencies

- Python development libraries.
- [pybind11](https://github.com/pybind/pybind11) for building Python bindings.

### Other dependencies

- [Doxygen](https://github.com/doxygen/doxygen) for building the documentation.
- [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css) the documentation theme.
- [argparse](https://github.com/p-ranav/argparse) for command-line argument parsing.
- Python packages
  - [graphviz](https://github.com/xflr6/graphviz) for model visualization
  - [pytest](https://docs.pytest.org/en/stable/index.html) for testing Pythin bindings
  - [PyYAML](https://pyyaml.org/) for extracting syntax documentation
  - [pybind11-stubgen](https://github.com/sizmailov/pybind11-stubgen) for extracting stubs from Python bindings
