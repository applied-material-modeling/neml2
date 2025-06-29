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
# Build types
# ----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage" , "ThreadSanitizer" , "Profiling")

# ----------------------------------------------------------------------------
# Enable gcov wrapper for clang
# ----------------------------------------------------------------------------
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
# Project-level settings, options, and flags
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${NEML2_SOURCE_DIR}/cmake/Modules)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by C++ compiler during coverage builds." FORCE)
set(CMAKE_CXX_FLAGS_THREADSANITIZER "-O0 -fsanitize=thread" CACHE STRING "Flags used by C++ compiler for build type ThreadSanitizer" FORCE)
set(CMAKE_CXX_FLAGS_PROFILING "-O2 -g -fno-omit-frame-pointer" CACHE STRING "Flags used by C++ compiler for build type Profiling" FORCE)
set(NEML2_PCH ON CACHE BOOL "Use precompiled headers")
set(NEML2_TESTS ON CACHE BOOL "Build NEML2 tests")
set(NEML2_RUNNER OFF CACHE BOOL "Build a simple runner for benchmarking, debugging, etc.")
set(NEML2_PYBIND OFF CACHE BOOL "Build NEML2 Python bindings")
set(NEML2_DOC OFF CACHE BOOL "Build NEML2 documentation (html)")
set(NEML2_WORK_DISPATCHER OFF CACHE BOOL "Enable NEML2 work dispatcher")
set(NEML2_JSON OFF CACHE BOOL "Enable JSON support")
set(AURORA_BUILD OFF CACHE BOOL "Build on ALCF Aurora machine")

# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
add_custom_target(deps)
set(TORCH_VERSION "2.5.1" CACHE STRING "Default libTorch/PyTorch version to download")
set(DOXYGEN_VERSION "1.13.2" CACHE STRING "Default doxygen version to download")
set(DOXYGEN_AWESOME_VERSION "2.3.4" CACHE STRING "Default doxygen-awesome version to download")
set(PYBIND11_VERSION "2.13.6" CACHE STRING "Default pybind11 version to download")
set(WASP_VERSION "cf46fbd0d224cc2a70edbc668d3e65d45bec9e16" CACHE STRING "Default WASP version to download (if not found)")
set(HIT_VERSION "2b56209d3f1a1c5f0bede15a3e13981b599a5f91" CACHE STRING "Default HIT version to download")
set(CATCH2_VERSION "3.5.4" CACHE STRING "Default Catch2 version to download (if not found)")
set(GPERFTOOLS_VERSION "2.15" CACHE STRING "Default gperftools version to download")
set(ARGPARSE_VERSION "3.0" CACHE STRING "Default argparse version to download")
set(TIMPI_VERSION "1.8.5_bootstrapped" CACHE STRING "Default TIMPI version to download")
set(NLOHMANN_JSON_VERSION "3.11.3" CACHE STRING "Default nlohmann json version to download")

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
# Torch
# ----------------------------------------------------------------------------
find_package(Torch)

if(NOT Torch_FOUND)
      include(DownloadTorch)
      set(Torch_ROOT ${torch_SOURCE_DIR} CACHE PATH "Torch root directory" FORCE)
      find_package(Torch REQUIRED)
endif()

if(HAS_GLIBC)
      if(${GLIBCXX_USE_CXX11_ABI} STREQUAL "0")
            message(STATUS "Using pre C++11 ABI")
      else()
            message(STATUS "Using C++11 ABI")
      endif()
endif()

cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${Torch_LINK_DIR} Torch_DOWNLOADED)

if(Torch_DOWNLOADED)
      install(DIRECTORY ${Torch_INCLUDE_DIR} TYPE INCLUDE COMPONENT libneml2)
      install(DIRECTORY ${Torch_LINK_DIR} DESTINATION . COMPONENT libneml2)
endif()

# ----------------------------------------------------------------------------
# Intel Extension for PyTorch
# ----------------------------------------------------------------------------
if (AURORA_BUILD)
      find_library(IPEX_LIB intel-ext-pt-gpu PATHS ${INTEL_EXTENSION_FOR_PYTORCH_PATH}/lib NO_DEFAULT_PATH REQUIRED)
endif()

# ----------------------------------------------------------------------------
# HIT
# ----------------------------------------------------------------------------
find_package(HIT)

if(NOT HIT_FOUND)
      # ----------------------------------------------------------------------------
      # WASP
      # ----------------------------------------------------------------------------
      if(NOT WASP_FOUND)
            include(DownloadWASP)
            set(WASP_ROOT ${WASP_INSTALL_DIR} CACHE PATH "WASP root directory" FORCE)
            find_package(WASP REQUIRED)
      endif()

      include(DownloadHIT)
      set(HIT_ROOT ${HIT_INSTALL_DIR} CACHE PATH "HIT root directory" FORCE)
      find_package(HIT REQUIRED)
endif()

cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${WASP_LINK_DIR} WASP_DOWNLOADED)
cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${HIT_LINK_DIR} HIT_DOWNLOADED)

if(WASP_DOWNLOADED)
      install(DIRECTORY ${WASP_INCLUDE_DIR}/waspcore ${WASP_INCLUDE_DIR}/wasphit TYPE INCLUDE COMPONENT libneml2)
endif()

if(HIT_DOWNLOADED)
      install(DIRECTORY ${HIT_INCLUDE_DIR}/hit TYPE INCLUDE COMPONENT libneml2)
endif()

# ----------------------------------------------------------------------------
# TIMPI
# ----------------------------------------------------------------------------
if(NEML2_WORK_DISPATCHER)
      find_package(TIMPI)

      if(NOT TIMPI_FOUND)
            include(DownloadTIMPI)
            set(TIMPI_ROOT ${TIMPI_INSTALL_DIR} CACHE PATH "TIMPI root directory" FORCE)
            find_package(TIMPI REQUIRED)
      endif()

      cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${TIMPI_LINK_DIR} TIMPI_DOWNLOADED)

      if(TIMPI_DOWNLOADED)
            install(DIRECTORY ${TIMPI_INCLUDE_DIR}/timpi TYPE INCLUDE COMPONENT libneml2)
            install(DIRECTORY ${TIMPI_LINK_DIR} DESTINATION . COMPONENT libneml2)
      endif()
endif()

# ----------------------------------------------------------------------------
# nlohmann json
# ----------------------------------------------------------------------------
if(NEML2_JSON)
      set(JSON_MultipleHeaders OFF)
      find_package(nlohmann_json QUIET)

      if(NOT TARGET nlohmann_json::nlohmann_json)
            include(FetchContent)
            FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v${NLOHMANN_JSON_VERSION}/json.tar.xz)
            message(STATUS "Downloading/updating nlohmann/json")
            set(JSON_Install ON)
            set(JSON_SystemInclude ON)
            FetchContent_MakeAvailable(json)
      endif()
endif()

# ----------------------------------------------------------------------------
# CPU Profiler
# ----------------------------------------------------------------------------
if(NEML2_TESTS OR NEML2_RUNNER)
      if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
            find_package(Gperftools)

            if(NOT Gperftools_FOUND)
                  include(DownloadGperftools)
                  set(Gperftools_ROOT ${GPERFTOOLS_INSTALL_DIR} CACHE PATH "gperftools root directory" FORCE)
                  find_package(Gperftools REQUIRED)
            endif()

            cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${Gperftools_LINK_DIR} Gperftools_DOWNLOADED)

            if(Gperftools_DOWNLOADED)
                  install(DIRECTORY ${Gperftools_LINK_DIR} DESTINATION . COMPONENT libneml2)
            endif()
      endif()
endif()

# ----------------------------------------------------------------------------
# Catch2
# ----------------------------------------------------------------------------
if(NEML2_TESTS)
      find_package(Catch2)

      if(NOT Catch2_FOUND)
            include(DownloadCatch2)
            set(Catch2_ROOT ${CATCH2_INSTALL_DIR} CACHE PATH "Catch2 root directory" FORCE)
            find_package(Catch2 REQUIRED)
      endif()
endif()

# ----------------------------------------------------------------------------
# Doxygen
# ----------------------------------------------------------------------------
if(NEML2_DOC AND NOT NEML2_PYBIND)
      message(WARNING "Documentation requires Python bindings (NEML2_PYBIND=ON), reverting NEML2_DOC to OFF")
      set(NEML2_DOC OFF CACHE BOOL "Build NEML2 documentation (html)" FORCE)
endif()

if(NEML2_DOC)
      find_package(Doxygen COMPONENTS dot)

      if(NOT DOXYGEN_FOUND)
            include(DownloadDoxygen)

            if(doxygen_POPULATED)
                  set(Doxygen_ROOT ${doxygen_SOURCE_DIR} CACHE PATH "Path to Doxygen installation" FORCE)
                  find_package(Doxygen COMPONENTS dot)
            endif()
      endif()

      if(DOXYGEN_FOUND)
            get_target_property(DOXYGEN_EXECUTABLE Doxygen::doxygen IMPORTED_LOCATION)
      else()
            message(WARNING "Documentation requires Doxygen, but one cannot be located, reverting NEML2_DOC to OFF")
            set(NEML2_DOC OFF CACHE BOOL "Build NEML2 documentation (html)" FORCE)
      endif()
endif()

# ----------------------------------------------------------------------------
# Doxygen html stylesheet
# ----------------------------------------------------------------------------
if(NEML2_DOC)
      include(FetchContent)
      FetchContent_Declare(
            doxygen-awesome-css
            GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
            GIT_TAG v${DOXYGEN_AWESOME_VERSION}
      )
      FetchContent_MakeAvailable(doxygen-awesome-css)
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
if(CMAKE_EXPORT_COMPILE_COMMANDS)
      file(CREATE_LINK ${NEML2_BINARY_DIR}/compile_commands.json ${NEML2_SOURCE_DIR}/compile_commands.json SYMBOLIC)
endif()
