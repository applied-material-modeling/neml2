# A standalone project that consumes an *installed* NEML2 exactly as a downstream
# C++ user would: find_package(neml2) pulls the shipped CMake package config, and
# the consumer links the exported neml2::aoti target. Driven by the `downstream`
# job in .github/workflows/cpp.yaml against a pip-installed wheel. NOT part of the
# main neml2 build (it is never add_subdirectory'd).
cmake_minimum_required(VERSION 3.26)
project(neml2_downstream LANGUAGES CXX)

find_package(neml2 REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE neml2::aoti)
