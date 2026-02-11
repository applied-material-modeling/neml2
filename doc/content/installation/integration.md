# External Project Integration {#external-project-integration}

[TOC]

## Makefile integration

Once NEML2 is installed, external C++ projects can use NEML2 by including the development headers and linking against the NEML2 libraries. NEML2 offers a collection of libraries, they are
- `neml2_base`
- `neml2_dispatcher`
- `neml2_driver`
- `neml2_misc`
- `neml2_model`
- `neml2_equation_system`
- `neml2_solver`
- `neml2_tensor`
- `neml2_user_tensor`

The names of the libraries are self-explanatory. As a general rule of thumb, the library names collide with the header hierarchy. That is, if your code includes `#include "neml2/tensors/SR2.h"`, then your program should be linked against the corresponding tensor library using `-lneml2_tensor`, etc. 

## CMake integration

### Sub-directory

Integrating NEML2 into a project that already uses CMake is fairly straightforward. The following CMakeLists.txt snippet links NEML2 into the target executable called `foo`:

```
add_subdirectory(neml2)
add_executable(foo main.cxx)
target_link_libraries(foo neml2)
```

The above snippet assumes NEML2 is checked out to the directory %neml2, i.e., as a git submodule.

### FetchContent

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

### find_package

NEML2 can also be discovered from its installation location relying on CMake's config mode `find_package` function

```
find_package(neml2 CONFIG)
add_executable(foo main.cxx)
target_link_libraries(foo neml2::neml2)
```

Note that the config mode search defines several imported targets under the `neml2::` namespace. For example, `neml2::misc` corresponds to the `misc` library of NEML2, etc. The `neml2::neml2` is an interface target that transitively links to all NEML2 libraries.
