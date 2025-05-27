# dependencies: cmake.version_min
cmake_minimum_required(VERSION 3.26)
project(FOO)
add_subdirectory(neml2)
add_executable(foo main.cxx)
target_link_libraries(foo PRIVATE neml2)
