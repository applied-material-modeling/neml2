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

## List of dependencies

C++ backend:

<!-- dependencies: torch.version -->
- [PyTorch](https://pytorch.org/get-started/locally/), version 2.5.1.
- [WASP](https://code.ornl.gov/neams-workbench/wasp) as the lexing and parsing backend for HIT.
- Testing:
  - [Catch2](https://github.com/catchorg/Catch2) for unit and regression testing.

The Runner:

- [argparse](https://github.com/p-ranav/argparse) for command-line argument parsing.
- Profiling:
  - [Gperftools](https://github.com/gperftools/gperftools) for profiling purposes.

Python package:

- [Python development libraries](https://docs.python.org/3/extending/extending.html) for python bindings.
- [pybind11](https://github.com/pybind/pybind11) for building Python bindings.
- [pybind11-stubgen](https://github.com/sizmailov/pybind11-stubgen) for extracting stubs from Python bindings.
- [pyzag](https://github.com/applied-material-modeling/pyzag) for training material models.
- Testing:
  - [pytest](https://docs.pytest.org/en/stable/index.html) for testing Pythin bindings.

Documentation:

- [Doxygen](https://github.com/doxygen/doxygen) for building the documentation.
- [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css) the documentation theme.
- [graphviz](https://github.com/xflr6/graphviz) for model visualization.
- [PyYAML](https://pyyaml.org/) for extracting syntax documentation.

Work dispatcher:

- [TIMPI](https://github.com/libMesh/TIMPI) for coordinating parallel workers.
- [json](https://github.com/nlohmann/json) for outputting event traces.

In addition to standard system library locations, the CMake configure script also searches for an installed torch Python package. Recent PyTorch releases within a few minor versions are likely to be compatible.

\warning
If no PyTorch is found after searching, a CPU-only libtorch binary is downloaded from the official website. Such libtorch is not able to use the CUDA co-processor, even if there is one. If using CUDA is desired, please install the torch Python package or a CUDA-enabled libtorch before configuring NEML2.

\note
We strive to keep up with the rapid development of PyTorch. The NEML2 PyTorch dependency is updated on a quarterly basis. If there is a particular version of PyTorch you'd like to use which is found to be incompatible with NEML2, please feel free to [create an issue](https://github.com/applied-material-modeling/neml2/issues).

## Skipping dependencies

In some cases, certain dependencies cannot be obtained or are incompatible with the build system, and it becomes desirable to keep using NEML2 with some capabilities disabled.

The following table summarizes the configure options that determine when a dependency is required, and hence how a dependency can be skipped.

| Dependency                   | Dependent configure option(s)                 |
| :--------------------------- | :-------------------------------------------- |
| Torch                        |                                               |
| WASP                         |                                               |
| Catch2                       | NEML2_TESTS                                   |
| argparse                     | NEML2_RUNNER                                  |
| Gperftools                   | NEML2_RUNNER && CMAKE_BUILD_TYPE == Profiling |
| Python development libraries | NEML2_PYBIND                                  |
| pybind11                     | NEML2_PYBIND                                  |
| pybind11-stubgen             | NEML2_PYBIND                                  |
| pyzag                        | NEML2_PYBIND && NEML2_TESTS                   |
| pytest                       | NEML2_PYBIND && NEML2_TESTS                   |
| Doxygen                      | NEML2_DOC                                     |
| Doxygen Awesome              | NEML2_DOC                                     |
| graphviz                     | NEML2_PYBIND && NEML2_TESTS                   |
| PyYAML                       | NEML2_DOC                                     |
| MPI, TIMPI                   | NEML2_WORK_DISPATCHER                         |
| json                         | NEML2_JSON                                    |
