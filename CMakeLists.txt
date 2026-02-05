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

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage" "ThreadSanitizer" "Profiling")

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
set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by C++ compiler during coverage builds" FORCE)
set(CMAKE_CXX_FLAGS_THREADSANITIZER "-O1 -g -fsanitize=thread" CACHE STRING "Flags used by C++ compiler for build type ThreadSanitizer" FORCE)
set(CMAKE_CXX_FLAGS_ADDRESSSANITIZER "-O1 -g -fsanitize=address -fno-omit-frame-pointer" CACHE STRING "Flags used by C++ compiler for build type AddressSanitizer" FORCE)
set(CMAKE_CXX_FLAGS_PROFILING "-O2 -g -fno-omit-frame-pointer -DNDEBUG" CACHE STRING "Flags used by C++ compiler for build type Profiling" FORCE)
set(NEML2_PCH ON CACHE BOOL "Use precompiled headers")
set(NEML2_TESTS ON CACHE BOOL "Build NEML2 tests")
set(NEML2_TOOLS OFF CACHE BOOL "Build utility binaries")
set(NEML2_WORK_DISPATCHER OFF CACHE BOOL "Enable NEML2 work dispatcher")
set(NEML2_JSON OFF CACHE BOOL "Enable JSON support")
set(NEML2_CSV OFF CACHE BOOL "Enable CSV support")
set(NEML2_CONTRIB_PREFIX ${NEML2_SOURCE_DIR}/contrib CACHE PATH "NEML2 contrib prefix for downloaded dependencies")
set(NEML2_CONTRIB_PARALLEL 1 CACHE STRING "Number of parallel jobs for NEML2 dependencies build")
set(NEML2_WHEEL OFF CACHE INTERNAL "Build NEML2 as a Python wheel. This is supposed to be set by setup.py and not by the user.")

# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
set(torch_VERSION "2.8.0" CACHE STRING "Default libTorch/PyTorch version to download")
set(torch_SEARCH_SITE_PACKAGES ON CACHE BOOL "Search for libTorch in Python site-packages")
set(wasp_VERSION "78e8705d3743cee12cb0b024a51a98db2c3713bc" CACHE STRING "Default WASP version to download (if not found)")
set(hit_VERSION "df7e49e95e5ea9ebef0e7578408221d3aeaf7513" CACHE STRING "Default HIT version to download")
set(catch2_VERSION "v3.5.4" CACHE STRING "Default Catch2 version to download (if not found)")
set(gperftools_VERSION "gperftools-2.17" CACHE STRING "Default gperftools version to download")
set(argparse_VERSION "v3.2" CACHE STRING "Default argparse version to download")
set(nlohmann_json_VERSION "v3.11.3" CACHE STRING "Default nlohmann json version to download")
set(csvparser_VERSION "2.3.0" CACHE STRING "Default csv-parser version to download")

# ----------------------------------------------------------------------------
# Install message
# ----------------------------------------------------------------------------
set(CMAKE_INSTALL_MESSAGE LAZY)

# ----------------------------------------------------------------------------
# For relocatable install
# ----------------------------------------------------------------------------
if(UNIX AND APPLE)
      set(INSTALL_REL_PATH "@loader_path")
elseif(UNIX AND NOT APPLE)
      set(INSTALL_REL_PATH "$ORIGIN")
endif()

# ----------------------------------------------------------------------------
# Utilities for downloading and installing dependencies
# ----------------------------------------------------------------------------
include(DepUtils)

# ----------------------------------------------------------------------------
# lib directory
# ----------------------------------------------------------------------------
# During an editable wheel install, we want to put the cpython libraries in the source tree so
# that they can be found by the redirected imports..
if(NEML2_WHEEL AND DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable")
      set(INSTALL_LIBDIR ${NEML2_SOURCE_DIR}/python/neml2/lib)
      set(INSTALL_BINDIR ${NEML2_SOURCE_DIR}/python/neml2/bin)
else()
      set(INSTALL_LIBDIR lib)
      set(INSTALL_BINDIR bin)
endif()


# ----------------------------------------------------------------------------
# The following blocks are used to find and download 3rd party dependencies
#
# We make some decent efforts to download and install the dependencies
# if they are not found. If they are found, we use the existing installation.
#
# We handle the dependencies a bit differently based on whether they are
# development dependencies.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Torch
# ----------------------------------------------------------------------------
find_package(torch MODULE OPTIONAL_COMPONENTS cuda python)

if(NOT torch_FOUND)
      if(UNIX)
            if(NOT APPLE)
                  set(torch_URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${torch_VERSION}%2Bcpu.zip)
            else()
                  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
                        set(torch_URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${torch_VERSION}.zip)
                  else()
                        set(torch_URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${torch_VERSION}.zip)
                  endif()
            endif()
      endif()

      if(torch_URL)
            download_from_url(torch ${torch_URL} ${NEML2_CONTRIB_PREFIX})
            set(torch_ROOT ${NEML2_CONTRIB_PREFIX}/torch-src)
            find_package(torch MODULE REQUIRED OPTIONAL_COMPONENTS cuda python)
      else()
            message(FATAL_ERROR "Torch not found.")
      endif()
endif()

# Note the torch libraries are never installed as part of the NEML2 installation.
#
# For a standard NEML2 installation, we add the absolute link path to the torch
# libraries as rpath.
#
# For a wheel installation, we prepend the rpaths
#   ../torch/lib
#   ../../torch/lib
# assuming the wheel is installed in a standard site-packages location alongside torch.

# find_package(neml2 CONFIG) will rely on the Findtorch.cmake module
if(NOT SKBUILD_STATE STREQUAL "editable")
      install(FILES
            ${NEML2_SOURCE_DIR}/cmake/Modules/Findtorch.cmake
            ${NEML2_SOURCE_DIR}/cmake/Modules/DetectTorchCXXABI.cxx
            DESTINATION share/cmake/neml2/Modules
            COMPONENT libneml2
      )
endif()

# ----------------------------------------------------------------------------
# wasp
# ----------------------------------------------------------------------------
find_package(wasp MODULE COMPONENTS core hit)

if(NOT wasp_FOUND)
      download_from_git(wasp https://code.ornl.gov/neams-workbench/wasp.git ${NEML2_CONTRIB_PREFIX} ${wasp_VERSION})
      set(wasp_INSTALL_PREFIX ${NEML2_CONTRIB_PREFIX}/wasp CACHE PATH "wasp install prefix")
      custom_install(wasp contrib/install_wasp.sh ${NEML2_CONTRIB_PREFIX}/wasp-src ${NEML2_CONTRIB_PREFIX}/wasp-build ${wasp_INSTALL_PREFIX})
      find_package(wasp MODULE REQUIRED COMPONENTS core hit)
endif()

# check if wasp is downloaded
path_has_prefix(${wasp_ROOT} ${NEML2_CONTRIB_PREFIX} wasp_CONTRIB)

# wasp will be packaged as part of the NEML2 installation if it was downloaded by us (or if it is being built as a wheel).
if(wasp_CONTRIB OR NEML2_WHEEL)
      install_glob(${wasp_core_LIBRARY} ${INSTALL_LIBDIR} libneml2)
      install_glob(${wasp_hit_LIBRARY} ${INSTALL_LIBDIR} libneml2)
      if(NOT SKBUILD_STATE STREQUAL "editable")
            install(DIRECTORY ${wasp_core_INCLUDE_DIR}/waspcore ${wasp_hit_INCLUDE_DIR}/wasphit TYPE INCLUDE COMPONENT libneml2)
      endif()
endif()
if(NOT SKBUILD_STATE STREQUAL "editable")
      install(FILES ${NEML2_SOURCE_DIR}/cmake/Modules/Findwasp.cmake DESTINATION share/cmake/neml2/Modules COMPONENT libneml2)
endif()

# ----------------------------------------------------------------------------
# hit
# ----------------------------------------------------------------------------
find_package(hit MODULE)

if(NOT hit_FOUND)
      if(NOT hit_SOURCE_DIR)
            download_from_git(hit https://github.com/idaholab/hit.git ${NEML2_CONTRIB_PREFIX} ${hit_VERSION})
            set(hit_SOURCE_DIR ${NEML2_CONTRIB_PREFIX}/hit-src)
      endif()

      file(COPY ${hit_SOURCE_DIR}/include DESTINATION ${NEML2_CONTRIB_PREFIX}/hit-build FILES_MATCHING PATTERN "*.h")
      file(COPY ${hit_SOURCE_DIR}/src DESTINATION ${NEML2_CONTRIB_PREFIX}/hit-build FILES_MATCHING PATTERN "*.cpp" PATTERN "*.cc")
      configure_file(contrib/install_hit.cmake ${NEML2_CONTRIB_PREFIX}/hit-build/CMakeLists.txt @ONLY)
      set(hit_INSTALL_PREFIX ${NEML2_CONTRIB_PREFIX}/hit CACHE PATH "hit install prefix")
      custom_install(hit contrib/install_hit.sh ${NEML2_CONTRIB_PREFIX}/hit-build ${NEML2_CONTRIB_PREFIX}/hit-build ${hit_INSTALL_PREFIX})
      find_package(hit MODULE REQUIRED)
endif()

# check if hit is downloaded
path_has_prefix(${hit_ROOT} ${NEML2_CONTRIB_PREFIX} hit_CONTRIB)

# hit will be packaged as part of the NEML2 installation if it was downloaded by us (or if it is being built as a wheel).
if(hit_CONTRIB OR NEML2_WHEEL)
      install(FILES ${hit_LIBRARY} DESTINATION ${INSTALL_LIBDIR} COMPONENT libneml2)
      if(NOT SKBUILD_STATE STREQUAL "editable")
            install(DIRECTORY ${hit_INCLUDE_DIR}/hit TYPE INCLUDE COMPONENT libneml2)
      endif()
endif()
if(NOT SKBUILD_STATE STREQUAL "editable")
      install(FILES ${NEML2_SOURCE_DIR}/cmake/Modules/Findhit.cmake DESTINATION share/cmake/neml2/Modules COMPONENT libneml2)
endif()

# ----------------------------------------------------------------------------
# nlohmann json
# ----------------------------------------------------------------------------
if(NEML2_JSON)
      find_package(nlohmann_json CONFIG HINTS ${NEML2_CONTRIB_PREFIX}/nlohmann_json)

      if(NOT nlohmann_json_FOUND)
            download_from_git(nlohmann_json https://github.com/nlohmann/json.git ${NEML2_CONTRIB_PREFIX} ${nlohmann_json_VERSION})
            set(nlohmann_json_INSTALL_PREFIX ${NEML2_CONTRIB_PREFIX}/nlohmann_json CACHE PATH "nlohmann json install prefix")
            custom_install(nlohmann_json contrib/install_nlohmann_json.sh ${NEML2_CONTRIB_PREFIX}/nlohmann_json-src ${NEML2_CONTRIB_PREFIX}/nlohmann_json-build ${nlohmann_json_INSTALL_PREFIX})
            find_package(nlohmann_json CONFIG REQUIRED PATHS ${nlohmann_json_INSTALL_PREFIX} NO_DEFAULT_PATH)
      endif()
      file(REAL_PATH "../../../" nlohmann_json_DIR BASE_DIRECTORY ${nlohmann_json_DIR})

      # check if nlohmann json is downloaded
      path_has_prefix(${nlohmann_json_DIR} ${NEML2_CONTRIB_PREFIX} nlohmann_json_CONTRIB)

      # nlohmann json will be packaged as part of the NEML2 installation if it was downloaded by us (or if it is being built as a wheel).
      if(nlohmann_json_CONTRIB OR NEML2_WHEEL)
            if(NOT SKBUILD_STATE STREQUAL "editable")
                  install(DIRECTORY ${nlohmann_json_DIR}/include/nlohmann TYPE INCLUDE COMPONENT libneml2)
            endif()
      endif()
      if(NOT SKBUILD_STATE STREQUAL "editable")
            install(DIRECTORY ${nlohmann_json_DIR}/share/ DESTINATION share COMPONENT libneml2)
      endif()
endif()

# ----------------------------------------------------------------------------
# csv-parser
# ----------------------------------------------------------------------------
if(NEML2_CSV)
      find_package(csvparser MODULE)

      if(NOT csvparser_FOUND)
            download_from_git(csvparser https://github.com/vincentlaucsb/csv-parser.git ${NEML2_CONTRIB_PREFIX} ${csvparser_VERSION})
            set(csvparser_INSTALL_PREFIX ${NEML2_CONTRIB_PREFIX}/csvparser CACHE PATH "csv-parser install prefix")
            file(COPY ${NEML2_CONTRIB_PREFIX}/csvparser-src/single_include/csv.hpp DESTINATION ${NEML2_CONTRIB_PREFIX}/csvparser/csvparser)
            find_package(csvparser MODULE REQUIRED)
      endif()

      # check if csvparser is downloaded
      path_has_prefix(${csvparser_ROOT} ${NEML2_CONTRIB_PREFIX} csvparser_CONTRIB)

      # csv-parser will be packaged as part of the NEML2 installation if it was downloaded by us (or if it is being built as a wheel).
      if(csvparser_CONTRIB OR NEML2_WHEEL)
            if(NOT SKBUILD_STATE STREQUAL "editable")
                  install(DIRECTORY ${csvparser_ROOT}/csvparser/csvparser TYPE INCLUDE COMPONENT libneml2)
            endif()
      endif()
      if(NOT SKBUILD_STATE STREQUAL "editable")
            install(FILES ${NEML2_SOURCE_DIR}/cmake/Modules/Findcsvparser.cmake DESTINATION share/cmake/neml2/Modules COMPONENT libneml2)
      endif()
endif()

# ----------------------------------------------------------------------------
# CPU Profiler
# ----------------------------------------------------------------------------
if(NEML2_TOOLS)
      if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
            find_package(gperftools MODULE COMPONENTS profiler)

            if(NOT gperftools_FOUND)
                  download_from_git(gperftools https://github.com/gperftools/gperftools.git ${NEML2_CONTRIB_PREFIX} ${gperftools_VERSION})
                  execute_process(
                        COMMAND bash -c ./autogen.sh
                        WORKING_DIRECTORY ${NEML2_CONTRIB_PREFIX}/gperftools-src
                        OUTPUT_QUIET OUTPUT_FILE bootstrap.log
                        ERROR_QUIET ERROR_FILE bootstrap.err
                  )
                  set(gperftools_INSTALL_PREFIX ${NEML2_CONTRIB_PREFIX}/gperftools CACHE PATH "gperftools install prefix")
                  custom_install(gperftools contrib/install_gperftools.sh ${NEML2_CONTRIB_PREFIX}/gperftools-src ${NEML2_CONTRIB_PREFIX}/gperftools-build ${gperftools_INSTALL_PREFIX})
                  find_package(gperftools MODULE REQUIRED COMPONENTS profiler)
            endif()

            # check if gperftools is downloaded
            path_has_prefix(${gperftools_ROOT} ${NEML2_CONTRIB_PREFIX} gperftools_CONTRIB)

            # gperftools will be packaged as part of the NEML2 installation if it was downloaded by us (or if it is being built as a wheel).
            if(gperftools_CONTRIB OR NEML2_WHEEL)
                  install_glob(${gperftools_profiler_LIBRARY} ${INSTALL_LIBDIR} libneml2)
            endif()
            if(NOT SKBUILD_STATE STREQUAL "editable")
                  install(FILES ${NEML2_SOURCE_DIR}/cmake/Modules/Findgperftools.cmake DESTINATION share/cmake/neml2/Modules COMPONENT libneml2)
            endif()
      endif()
endif()

# ----------------------------------------------------------------------------
# Catch2
# ----------------------------------------------------------------------------
if(NEML2_TESTS)
      include(CTest)
      find_package(Catch2 CONFIG HINTS ${NEML2_CONTRIB_PREFIX}/Catch2)

      if(NOT Catch2_FOUND)
            download_from_git(Catch2 https://github.com/catchorg/Catch2.git ${NEML2_CONTRIB_PREFIX} ${catch2_VERSION})
            set(Catch2_INSTALL_PREFIX ${NEML2_CONTRIB_PREFIX}/Catch2 CACHE PATH "Catch2 install prefix")
            custom_install(Catch2 contrib/install_Catch2.sh ${NEML2_CONTRIB_PREFIX}/Catch2-src ${NEML2_CONTRIB_PREFIX}/Catch2-build ${Catch2_INSTALL_PREFIX})
            find_package(Catch2 CONFIG REQUIRED PATHS ${Catch2_INSTALL_PREFIX} NO_DEFAULT_PATH)
      endif()
endif()

# ----------------------------------------------------------------------------
# MPI
# ----------------------------------------------------------------------------
if(NEML2_WORK_DISPATCHER)
      find_package(MPI COMPONENTS CXX REQUIRED)
endif()

# ----------------------------------------------------------------------------
# argparse
# ----------------------------------------------------------------------------
if(NEML2_TOOLS)
      find_package(argparse CONFIG HINTS ${NEML2_CONTRIB_PREFIX}/argparse)

      if(NOT argparse_FOUND)
            download_from_git(argparse https://github.com/p-ranav/argparse.git ${NEML2_CONTRIB_PREFIX} ${argparse_VERSION})
            set(argparse_INSTALL_PREFIX ${NEML2_CONTRIB_PREFIX}/argparse CACHE PATH "argparse install prefix")
            custom_install(argparse contrib/install_argparse.sh ${NEML2_CONTRIB_PREFIX}/argparse-src ${NEML2_CONTRIB_PREFIX}/argparse-build ${argparse_INSTALL_PREFIX})
            find_package(argparse CONFIG REQUIRED PATHS ${argparse_INSTALL_PREFIX} NO_DEFAULT_PATH)
      endif()
endif()

# ----------------------------------------------------------------------------
# base neml2 library
# ----------------------------------------------------------------------------
add_subdirectory(src/neml2)

# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
if(NEML2_TESTS)
      add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# tools
# ----------------------------------------------------------------------------
if(NEML2_TOOLS)
      add_subdirectory(src/tools)
endif()

# ----------------------------------------------------------------------------
# Python bindings
# ----------------------------------------------------------------------------
if(NEML2_WHEEL)
      add_subdirectory(python)
endif()

# ----------------------------------------------------------------------------
# compile_commands.json
# ----------------------------------------------------------------------------
if(CMAKE_EXPORT_COMPILE_COMMANDS)
      set(SYMLINK_NAME "${NEML2_SOURCE_DIR}/compile_commands.json")
      set(FILE_ORIGINAL "${NEML2_BINARY_DIR}/compile_commands.json")

      if(NOT ${SYMLINK_NAME} STREQUAL ${FILE_ORIGINAL})
            file(CREATE_LINK ${NEML2_BINARY_DIR}/compile_commands.json ${NEML2_SOURCE_DIR}/compile_commands.json SYMBOLIC)
      endif()
endif()
