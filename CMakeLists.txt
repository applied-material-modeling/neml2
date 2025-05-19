# ----------------------------------------------------------------------------
# Project metadata
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.26)
project(NEML2 VERSION 2.0.0 LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------
# FindPython should return the first matching Python
if(POLICY CMP0094)
      cmake_policy(SET CMP0094 NEW)
endif()

# Suppress the warning related to the new policy on fetch content's timestamp
if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
endif()

# Suppress the warning related to the new policy on FindPythonXXX
if(POLICY CMP0148)
      cmake_policy(SET CMP0148 NEW)
endif()

# ----------------------------------------------------------------------------
# Project-level settings, options, and flags
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${NEML2_SOURCE_DIR}/cmake/Modules)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by C++ compiler during coverage builds." FORCE)
set(CMAKE_CXX_FLAGS_THREAD_SANITIZER "-fsanitize=thread" CACHE STRING "Flags used by ThreadSanitizer" FORCE)
set(NEML2_PCH ON CACHE BOOL "Use precompiled headers")
set(NEML2_TESTS ON CACHE BOOL "Build NEML2 tests")
set(NEML2_RUNNER OFF CACHE BOOL "Build a simple runner for benchmarking, profiling, debugging, etc.")
set(NEML2_PYBIND OFF CACHE BOOL "Build NEML2 Python bindings")
set(NEML2_DOC OFF CACHE BOOL "Build NEML2 documentation (html)")
set(NEML2_CPU_PROFILER OFF CACHE BOOL "Link against gperftools profiler to enable CPU profiling")
set(NEML2_WORK_DISPATCHER OFF CACHE BOOL "Enable NEML2 work dispatcher")
set(NEML2_THREAD_SANITIZER OFF CACHE BOOL "Enable ThreadSanitizer")
set(NEML2_JSON OFF CACHE BOOL "Enable JSON support")

# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
add_custom_target(deps)
set(TORCH_VERSION "2.5.1" CACHE STRING "Default libTorch/PyTorch version to download")
set(DOXYGEN_VERSION "1.13.2" CACHE STRING "Default doxygen version to download")
set(DOXYGEN_AWESOME_VERSION "2.3.4" CACHE STRING "Default doxygen-awesome version to download")
set(PYBIND11_VERSION "2.13.6" CACHE STRING "Default pybind11 version to download")
set(WASP_VERSION "c8c9ce4259115973f147e345608450d87dc390c4" CACHE STRING "Default WASP version to download (if not found)")
set(HIT_VERSION "2b56209d3f1a1c5f0bede15a3e13981b599a5f91" CACHE STRING "Default HIT version to download")
set(CATCH2_VERSION "3.5.4" CACHE STRING "Default Catch2 version to download (if not found)")
set(GPERFTOOLS_VERSION "2.15" CACHE STRING "Default gperftools version to download")
set(ARGPARSE_VERSION "3.0" CACHE STRING "Default argparse version to download")
set(TIMPI_VERSION "1.8.5_bootstrapped" CACHE STRING "Default TIMPI version to download")
set(NLOHMANN_JSON_VERSION "3.11.3" CACHE STRING "Default nlohmann json version to download")

# ----------------------------------------------------------------------------
# Build types
# ----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage")

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
      if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(GCOV_TOOL "${NEML2_SOURCE_DIR}/scripts/gcov_clang_wrapper.sh")
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(GCOV_TOOL "gcov")
      else()
            message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID} for coverage build")
      endif()

      set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)
      configure_file(scripts/coverage.sh.in ${NEML2_BINARY_DIR}/scripts/coverage.sh)
endif()

# ----------------------------------------------------------------------------
# Install message
# ----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_INSTALL_MESSAGE)
      set(CMAKE_INSTALL_MESSAGE LAZY)
endif()

# ----------------------------------------------------------------------------
# For relocatable install
# ----------------------------------------------------------------------------
if(UNIX AND APPLE)
      set(EXEC_DIR "@loader_path")
elseif(UNIX AND NOT APPLE)
      set(EXEC_DIR "$ORIGIN")
else()
      install(CODE "message(WARNING \"The NEML2 installation may not be relocatable, and LD_LIBRARY_PATH should be used to dynamically load libraries\")")
endif()

if(DEFINED EXEC_DIR)
      include(GNUInstallDirs)
      file(RELATIVE_PATH LIB_REL_PATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR} ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
      set(CMAKE_INSTALL_RPATH "${EXEC_DIR};${EXEC_DIR}/${LIB_REL_PATH};${EXEC_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE STRING "Install RPATH")
endif()

# ----------------------------------------------------------------------------
# base neml2 library
# ----------------------------------------------------------------------------
include(CTest)

# Add the base library and its dependencies
add_subdirectory(src/neml2)

# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
if(NEML2_TESTS)
      add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------
if(NEML2_RUNNER)
      add_subdirectory(runner)
endif()

# ----------------------------------------------------------------------------
# Python bindings
# ----------------------------------------------------------------------------
if(NEML2_PYBIND)
      add_subdirectory(python)
endif()

# ----------------------------------------------------------------------------
# Documentation
# ----------------------------------------------------------------------------
if(NEML2_DOC)
      add_subdirectory(doc)
endif()

# ----------------------------------------------------------------------------
# compile_commands.json
# ----------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      file(CREATE_LINK ${NEML2_BINARY_DIR}/compile_commands.json ${NEML2_SOURCE_DIR}/compile_commands.json SYMBOLIC)
endif()
