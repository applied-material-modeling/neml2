# ----------------------------------------------------------------------------
# Project metadata
# ----------------------------------------------------------------------------
# dependencies: cmake.version_min
cmake_minimum_required(VERSION 3.26.1)
# dependencies: neml2.version
project(NEML2 VERSION 3.0.3 LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# Testing (enable_testing + ctest plumbing)
# ----------------------------------------------------------------------------
# include(CTest) sets BUILD_TESTING ON by default and creates ctest entries.
# The benchmark suite (the only ctest consumer today) is registered under
# benchmark/CMakeLists.txt and gated on this variable + NEML2_WHEEL=OFF so
# wheel builds skip it.
include(CTest)

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

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

# ----------------------------------------------------------------------------
# Project-level settings, options, and flags
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${NEML2_SOURCE_DIR}/cmake/Modules)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NEML2_CONTRIB_PREFIX ${NEML2_SOURCE_DIR}/contrib CACHE PATH "NEML2 contrib prefix for downloaded dependencies")
set(NEML2_WHEEL OFF CACHE INTERNAL "Build NEML2 as a Python wheel. This is supposed to be set by setup.py and not by the user.")

# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
set(torch_SEARCH_SITE_PACKAGES ON CACHE BOOL "Search for libTorch in Python site-packages")

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
# During an editable wheel install, we want to put the cpython libraries in the
# source tree so that they can be found by the redirected imports.
if(NEML2_WHEEL AND DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable")
      set(INSTALL_LIBDIR ${NEML2_SOURCE_DIR}/neml2/lib)
      set(INSTALL_BINDIR ${NEML2_SOURCE_DIR}/neml2/bin)
else()
      set(INSTALL_LIBDIR lib)
      set(INSTALL_BINDIR bin)
endif()

# ----------------------------------------------------------------------------
# Torch
# ----------------------------------------------------------------------------
find_package(torch MODULE OPTIONAL_COMPONENTS cuda python)

if(NOT torch_FOUND)
      message(FATAL_ERROR "Torch not found.")
endif()

# ----------------------------------------------------------------------------
# AOTInductor header check
# ----------------------------------------------------------------------------
# torch::inductor::AOTIModelPackageLoader lives in
# torch/csrc/inductor/aoti_package/model_package_loader.h. The header exists in
# libtorch ≥ 2.5 (when AOTI moved to the package format). Verify it's present
# in the discovered libtorch before configuring the aoti submodule.
find_file(torch_AOTI_HEADER
      NAMES torch/csrc/inductor/aoti_package/model_package_loader.h
      PATHS ${torch_INCLUDE_DIR}
      NO_DEFAULT_PATH
)
if(NOT torch_AOTI_HEADER)
      message(FATAL_ERROR
            "model_package_loader.h was not found in the discovered libtorch "
            "(${torch_INCLUDE_DIR}). The header ships with PyTorch ≥ 2.5 in "
            "torch/csrc/inductor/aoti_package/. Upgrade PyTorch."
      )
endif()

if(NOT SKBUILD_STATE STREQUAL "editable")
      install(FILES
            ${NEML2_SOURCE_DIR}/cmake/Modules/Findtorch.cmake
            ${NEML2_SOURCE_DIR}/cmake/Modules/DetectTorchCXXABI.cxx
            DESTINATION share/cmake/neml2/Modules
            COMPONENT libneml2
      )
endif()

# ----------------------------------------------------------------------------
# nlohmann json
# ----------------------------------------------------------------------------
# Sourced from the contrib/nlohmann_json-src git submodule and built+installed
# into contrib/nlohmann_json by contrib/install_nlohmann_json.sh. We auto-init
# the submodule on first configure to keep the dev experience close to
# "clone, configure, build". Pinned via the submodule SHA.
find_package(nlohmann_json CONFIG HINTS ${NEML2_CONTRIB_PREFIX}/nlohmann_json)

if(NOT nlohmann_json_FOUND)
      if(NOT EXISTS ${NEML2_SOURCE_DIR}/contrib/nlohmann_json-src/CMakeLists.txt)
            message(STATUS "nlohmann_json not found, checking out submodule...")
            execute_process(
                  COMMAND git submodule update --init --recursive -- contrib/nlohmann_json-src
                  WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
                  RESULT_VARIABLE nlohmann_json_submodule_result
                  OUTPUT_VARIABLE nlohmann_json_submodule_output
                  ERROR_VARIABLE nlohmann_json_submodule_error
            )
            if(NOT nlohmann_json_submodule_result EQUAL 0)
                  message(FATAL_ERROR
                        "Failed to initialize the nlohmann_json submodule (exit code: ${nlohmann_json_submodule_result}).\n"
                        "git output:\n${nlohmann_json_submodule_output}\n"
                        "git error:\n${nlohmann_json_submodule_error}\n"
                        "Please ensure git is available and the source tree is a git checkout, or run:\n"
                        "  git submodule update --init --recursive -- contrib/nlohmann_json-src"
                  )
            endif()
      endif()
      set(nlohmann_json_INSTALL_PREFIX ${NEML2_CONTRIB_PREFIX}/nlohmann_json CACHE PATH "nlohmann json install prefix")
      custom_install(nlohmann_json contrib/install_nlohmann_json.sh ${NEML2_SOURCE_DIR}/contrib/nlohmann_json-src ${NEML2_CONTRIB_PREFIX}/nlohmann_json-build ${nlohmann_json_INSTALL_PREFIX})
      find_package(nlohmann_json CONFIG REQUIRED PATHS ${nlohmann_json_INSTALL_PREFIX} NO_DEFAULT_PATH)
endif()
file(REAL_PATH "../../../" nlohmann_json_DIR BASE_DIRECTORY ${nlohmann_json_DIR})

# check if nlohmann json is the in-tree (submodule-built) copy
path_has_prefix(${nlohmann_json_DIR} ${NEML2_CONTRIB_PREFIX} nlohmann_json_CONTRIB)

# nlohmann json is packaged with the NEML2 installation if we built it ourselves
# (submodule path) or if this is a wheel build.
if(nlohmann_json_CONTRIB OR NEML2_WHEEL)
      if(NOT SKBUILD_STATE STREQUAL "editable")
            install(DIRECTORY ${nlohmann_json_DIR}/include/nlohmann TYPE INCLUDE COMPONENT libneml2)
      endif()
endif()
if(NOT SKBUILD_STATE STREQUAL "editable")
      install(DIRECTORY ${nlohmann_json_DIR}/share/ DESTINATION share COMPONENT libneml2)
endif()

# ----------------------------------------------------------------------------
# libneml2_aoti — the only C++ artifact NEML2 ships
# ----------------------------------------------------------------------------
# Free-standing: links only against torch::core (+ torch::cuda when available)
# and nlohmann_json. Wraps torch::inductor::AOTIModelPackageLoader so the
# Python side can execute .pt2 artifacts produced by torch._inductor.aoti_compile_and_package
# from C++. Source tree lives at neml2/csrc/aoti/, intentionally co-located with
# the Python package so the C++ and Python sides of NEML2 stay in one tree.
# The single Model class is split across four translation units (construction,
# public ops, the value/Newton path, the Jacobian/IFT path); the internals live
# behind a PImpl `Model::Impl` declared in the non-shipped internal.h.
add_library(aoti SHARED
      neml2/csrc/aoti/Model.cpp
      neml2/csrc/aoti/ops.cpp
      neml2/csrc/aoti/solve.cpp
      neml2/csrc/aoti/jacobian.cpp
)

# API visibility control: hide everything by default and export only the symbols
# explicitly tagged with the generated AOTI_EXPORT macro (i.e. the public `Model`
# surface). This keeps the shipped ABI matched to the shipped header -- the
# PImpl internals + anonymous-namespace helpers stay out of the export table.
# Mirrors the hidden-visibility preset already used for the pybind module.
# Emit into a DEDICATED generated-include root (.../include), never the build-dir
# root: ${NEML2_BINARY_DIR} also holds stray files like `version`/`hash`, and
# putting it on the include path would shadow the C++ standard <version> header.
include(GenerateExportHeader)
generate_export_header(aoti
      BASE_NAME aoti
      EXPORT_FILE_NAME ${NEML2_BINARY_DIR}/include/neml2/csrc/aoti/aoti_export.h
)
set_target_properties(aoti PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
)

# BASE_DIRS at the project root means:
#  - sources include with `#include "neml2/csrc/aoti/Model.h"` (the file
#    location relative to the root); the BUILD_INTERFACE include dir is
#    ${NEML2_SOURCE_DIR}, so the path resolves
#  - on install the header lands at <prefix>/include/neml2/csrc/aoti/Model.h
#    and the imported target's INTERFACE_INCLUDE_DIRECTORIES is
#    <install-prefix>/include — downstream consumers using
#    find_package(neml2) write the same `#include "neml2/csrc/aoti/Model.h"`
#    without ever adding the wheel root to their include search path
target_sources(aoti
      PUBLIC
      FILE_SET HEADERS
      BASE_DIRS ${NEML2_SOURCE_DIR}
      FILES
      ${NEML2_SOURCE_DIR}/neml2/csrc/aoti/Model.h
)
# The generated aoti_export.h lives under the build tree's dedicated include
# root. Add it as a build-interface include so `#include
# "neml2/csrc/aoti/aoti_export.h"` resolves while compiling the lib + the pyaoti
# binding (which links aoti and inherits this). On install it is shipped beside
# Model.h (see the install() below), and the install-interface `include` dir
# already on the target covers it.
target_include_directories(aoti PUBLIC $<BUILD_INTERFACE:${NEML2_BINARY_DIR}/include>)
set_target_properties(aoti PROPERTIES OUTPUT_NAME "neml2_aoti$<IF:$<CONFIG:Release>,,_$<CONFIG>>")
target_compile_options(aoti PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(aoti PUBLIC torch::core nlohmann_json::nlohmann_json)
if(TARGET torch::cuda)
      target_link_libraries(aoti PUBLIC torch::cuda)
endif()

# rpath: only the torch hop is needed (no sibling neml2_*.so libraries to
# resolve anymore).
if(NEML2_WHEEL AND NOT SKBUILD_STATE STREQUAL "editable")
      set_target_properties(aoti PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/../../torch/lib")
else()
      set_target_properties(aoti PROPERTIES INSTALL_RPATH "${torch_LINK_DIR}")
endif()

install(TARGETS aoti
      LIBRARY DESTINATION ${INSTALL_LIBDIR}
      COMPONENT libneml2
)
if(NOT SKBUILD_STATE STREQUAL "editable")
      install(TARGETS aoti
            EXPORT neml2targets
            FILE_SET HEADERS DESTINATION include
            COMPONENT libneml2
      )
      # Ship the generated export header alongside Model.h. Editable installs
      # don't ship headers and resolve it via the build-tree include dir above.
      install(FILES ${NEML2_BINARY_DIR}/include/neml2/csrc/aoti/aoti_export.h
            DESTINATION include/neml2/csrc/aoti
            COMPONENT libneml2
      )
endif()

# ----------------------------------------------------------------------------
# Version / hash
# ----------------------------------------------------------------------------
find_package(Git)

file(WRITE ${NEML2_BINARY_DIR}/version "v${PROJECT_VERSION}\n")

if(Git_FOUND)
      execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
            OUTPUT_VARIABLE NEML2_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      file(WRITE ${NEML2_BINARY_DIR}/hash "${NEML2_HASH}\n")

      install(FILES
            ${NEML2_BINARY_DIR}/version
            ${NEML2_BINARY_DIR}/hash
            DESTINATION .
            COMPONENT libneml2
      )
endif()

# ----------------------------------------------------------------------------
# CMake package export (for downstream `find_package(neml2)`)
# ----------------------------------------------------------------------------
if(NOT SKBUILD_STATE STREQUAL "editable")
      install(EXPORT neml2targets NAMESPACE neml2:: DESTINATION share/cmake/neml2 COMPONENT libneml2)
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
      ${NEML2_SOURCE_DIR}/cmake/neml2Config.cmake.in
      ${NEML2_BINARY_DIR}/neml2Config.cmake
      INSTALL_DESTINATION share/cmake/neml2
      NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
      ${NEML2_BINARY_DIR}/neml2ConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMajorVersion
)

if(NOT SKBUILD_STATE STREQUAL "editable")
      install(
            FILES
            ${NEML2_BINARY_DIR}/neml2Config.cmake
            ${NEML2_BINARY_DIR}/neml2ConfigVersion.cmake
            DESTINATION share/cmake/neml2
            COMPONENT libneml2
      )
endif()

include(cmake/pkgconfig.cmake)

# ----------------------------------------------------------------------------
# Python bindings (pybind11 extensions co-located with the Python package)
# ----------------------------------------------------------------------------
if(NEML2_WHEEL)
      find_package(Python3 COMPONENTS Development.Module)
      get_target_property(torch_LINK_DIR torch::python INTERFACE_LINK_DIRECTORIES)

      # Top-level Python-bindings build target. Aggregating individual
      # pybind11 extensions under one custom target keeps the
      # `cmake --build --target pyneml2` entry point stable as new bindings
      # are added.
      add_custom_target(pyneml2)

      # ----------------------------------------------------------------------
      # add_native_extension(<target> <subpath> [LIBS lib1 lib2 ...])
      # ----------------------------------------------------------------------
      # Builds a pybind11 module from neml2/csrc/<subpath>/_<basename>.cpp
      # and outputs it at neml2/<subpath>/_<basename>.<soabi>.so so it's
      # importable as `neml2.<subpath>._<basename>`. <basename> = the last
      # component of <subpath>. Set ``LIBS`` to the C++ neml2 libraries the
      # binding links against (typically just `aoti`).
      macro(add_native_extension target subpath)
            set(_options)
            set(_oneValueArgs)
            set(_multiValueArgs LIBS)
            cmake_parse_arguments(NX "${_options}" "${_oneValueArgs}" "${_multiValueArgs}" ${ARGN})

            string(REGEX REPLACE "^.*/" "" _basename "${subpath}")
            set(_src "${NEML2_SOURCE_DIR}/neml2/csrc/${subpath}/_${_basename}.cpp")
            set(_outdir "neml2/${subpath}")

            # rpath hops from the .so to its dependency libraries. Both
            # destinations are expressed relative to the wheel root (the
            # site-packages dir for an installed wheel, the source tree for
            # an editable install). file(RELATIVE_PATH) does the .. math.
            #
            # - C++ NEML2 libs live at neml2/lib/        (under the wheel root)
            # - torch libs (wheel only) at torch/lib/    (sibling of neml2/)
            # - this .so lives at       neml2/<subpath>/
            file(RELATIVE_PATH _rel_libs "/neml2/${subpath}" "/neml2/lib")
            file(RELATIVE_PATH _rel_torch "/neml2/${subpath}" "/torch/lib")
            set(_rpath_to_libs "${INSTALL_REL_PATH}/${_rel_libs}")
            set(_rpath_to_torch "${INSTALL_REL_PATH}/${_rel_torch}")

            add_library(${target} MODULE ${_src})
            # Pybind sources include the C++ neml2 headers via the same
            # ``#include "neml2/csrc/..."`` path downstream consumers use;
            # `target_link_libraries(... aoti)` propagates aoti's BASE_DIRS
            # (NEML2_SOURCE_DIR) onto this target so the path resolves
            # without an extra target_include_directories.
            target_include_directories(${target} PUBLIC ${Python3_INCLUDE_DIRS})
            target_link_libraries(${target} PUBLIC torch::python Python3::Module ${NX_LIBS})
            target_compile_definitions(${target} PRIVATE "PYBIND11_DETAILED_ERROR_MESSAGES")
            set_target_properties(${target} PROPERTIES
                  LIBRARY_OUTPUT_DIRECTORY "${_outdir}"
                  OUTPUT_NAME "_${_basename}"
                  PREFIX ""
                  SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
                  CXX_VISIBILITY_PRESET hidden
            )
            if(DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable")
                  set_target_properties(${target} PROPERTIES INSTALL_RPATH "${_rpath_to_libs};${torch_LINK_DIR}")
            else()
                  set_target_properties(${target} PROPERTIES INSTALL_RPATH "${_rpath_to_libs};${_rpath_to_torch}")
            endif()
            if(CMAKE_BUILD_TYPE STREQUAL "Release")
                  set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
            endif()

            if(DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable")
                  install(TARGETS ${target} LIBRARY DESTINATION ${NEML2_SOURCE_DIR}/neml2/${subpath})
            else()
                  install(TARGETS ${target} LIBRARY DESTINATION "${subpath}")
            endif()
            add_dependencies(pyneml2 ${target})
      endmacro()

      add_native_extension(pyaoti aoti LIBS aoti)
endif()

# ----------------------------------------------------------------------------
# Benchmark suite (ctest entries; one per benchmark/<scenario>/model.i)
# ----------------------------------------------------------------------------
# Wheel builds skip ctest entries entirely — they're a developer convenience,
# not something shipped in the wheel. The dev preset (Debug, NEML2_WHEEL=OFF)
# is the consumer.
if(BUILD_TESTING AND NOT NEML2_WHEEL)
      add_subdirectory(benchmark)
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
