# CMake integration {#cmake-integration}

[TOC]

## Sub-directory

Integrating NEML2 into a project that already uses CMake is fairly straightforward. The following CMakeLists.txt snippet links NEML2 into the target executable called `foo`:

```
add_subdirectory(neml2)

add_executable(foo main.cxx)
target_link_libraries(foo neml2)
```

The above snippet assumes NEML2 is checked out to the directory %neml2, i.e., as a git submodule.

## FetchContent

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
