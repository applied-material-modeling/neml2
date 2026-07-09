# ----------------------------------------------------------------------------
# Project metadata
# ----------------------------------------------------------------------------
# dependencies: cmake.version_min
cmake_minimum_required(VERSION 3.26.1)
# dependencies: neml2.version
project(NEML2 VERSION 3.0.6 LANGUAGES C CXX)

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

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage" "ThreadSanitizer")

# The pip torch wheels ship Release (/MD) binaries only on Windows -- no debug
# import libs or debug DLLs -- so a Debug (/MDd) neml2 build cannot link or run
# against them. Reject single-config Debug on Windows with a clear message (use
# Release or RelWithDebInfo, both /MD). Multi-config generators pick the config
# at build time, so this configure-time check only covers single-config; a
# `--build --config Debug` there fails later at link, as expected.
get_property(_neml2_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(WIN32 AND NOT _neml2_multi_config AND CMAKE_BUILD_TYPE STREQUAL "Debug")
      message(FATAL_ERROR
            "Debug builds are unsupported on Windows: the pip torch wheels ship "
            "only Release (/MD) binaries. Configure with -DCMAKE_BUILD_TYPE=Release "
            "or -DCMAKE_BUILD_TYPE=RelWithDebInfo.")
endif()

# Instrumentation build types (opt in with -DCMAKE_BUILD_TYPE=<name>; clang or
# gcc). Selecting the build type *is* the opt-in -- they add no flags to a normal
# Debug/Release build and are off the wheel path (the wheel always builds
# Release), so shipped artifacts are never instrumented.
#   Coverage        -- clang source-based coverage. Drive it with
#                      scripts/cpp_coverage.sh (run tests -> profdata -> llvm-cov).
#   ThreadSanitizer -- data-race detector for the async dispatch pool. Run the
#                      C++ tests with the tests/cpp/tsan_suppressions.txt below.
set(CMAKE_C_FLAGS_COVERAGE "-O0 -g -fprofile-instr-generate -fcoverage-mapping"
      CACHE STRING "Flags used by the C compiler for Coverage builds" FORCE)
set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -g -fprofile-instr-generate -fcoverage-mapping"
      CACHE STRING "Flags used by the C++ compiler for Coverage builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "-fprofile-instr-generate"
      CACHE STRING "Linker flags for executables in Coverage builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "-fprofile-instr-generate"
      CACHE STRING "Linker flags for shared libraries in Coverage builds" FORCE)
set(CMAKE_C_FLAGS_THREADSANITIZER "-O1 -g -fsanitize=thread"
      CACHE STRING "Flags used by the C compiler for ThreadSanitizer builds" FORCE)
set(CMAKE_CXX_FLAGS_THREADSANITIZER "-O1 -g -fsanitize=thread"
      CACHE STRING "Flags used by the C++ compiler for ThreadSanitizer builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_THREADSANITIZER "-fsanitize=thread"
      CACHE STRING "Linker flags for executables in ThreadSanitizer builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_THREADSANITIZER "-fsanitize=thread"
      CACHE STRING "Linker flags for shared libraries in ThreadSanitizer builds" FORCE)
mark_as_advanced(
      CMAKE_C_FLAGS_COVERAGE CMAKE_CXX_FLAGS_COVERAGE
      CMAKE_EXE_LINKER_FLAGS_COVERAGE CMAKE_SHARED_LINKER_FLAGS_COVERAGE
      CMAKE_C_FLAGS_THREADSANITIZER CMAKE_CXX_FLAGS_THREADSANITIZER
      CMAKE_EXE_LINKER_FLAGS_THREADSANITIZER CMAKE_SHARED_LINKER_FLAGS_THREADSANITIZER)

# ----------------------------------------------------------------------------
# Project-level settings, options, and flags
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${NEML2_SOURCE_DIR}/cmake/Modules)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC-wide build flags (apply to every target below):
#  - /Zc:__cplusplus: MSVC otherwise reports __cplusplus == 199711L even under
#    /std:c++17, which trips standards-conditional code in torch's headers.
#  - /bigobj: the torch + typed-wrapper template instantiations blow past the
#    default per-object section limit (fatal C1128) without it.
#  - /external:W0: silence warnings from third-party headers (torch, nmhit,
#    nlohmann), which CMake marks as external (/external:I) for SYSTEM includes --
#    the MSVC equivalent of the automatic -isystem suppression on GCC/Clang. This
#    lets a warnings-as-errors build (CXXFLAGS=/WX in CI, mirroring -Werror on the
#    other platforms) not choke on warnings inside those headers.
#  - /wd4267 (size_t -> smaller-int narrowing): torch triggers this by passing a
#    size_t into a std::optional<int> (function_schema.h). It surfaces in the STL
#    header <optional> -- not in our external set, so /external:W0 can't reach it
#    -- and it is not part of our GCC/Clang -Wall/-Wextra baseline anyway (that
#    is -Wconversion, which we do not enable), so disabling it keeps the warning
#    surface consistent across platforms rather than erroring only on MSVC.
# And _CRT_SECURE_NO_WARNINGS: silence C4996 on standard calls like std::getenv.
if(MSVC)
      add_compile_options(/Zc:__cplusplus /bigobj /external:W0 /wd4267)
      add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

set(NEML2_CONTRIB_PREFIX ${NEML2_SOURCE_DIR}/contrib CACHE PATH "NEML2 contrib prefix for downloaded dependencies")
set(NEML2_WHEEL OFF CACHE INTERNAL "Build NEML2 as a Python wheel. This is supposed to be set by setup.py and not by the user.")

# MPI is optional and OFF by default: it powers only the MPISimpleScheduler (one GPU
# per rank). With the flag OFF the scheduler's constructor throws a clear
# "rebuild with -DNEML2_MPI=ON" error, so the class is always present.
option(NEML2_MPI "Build the MPI work scheduler (links against MPI)" OFF)

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
# Install directories
# ----------------------------------------------------------------------------
# An editable install builds into ``build/<wheel_tag>`` but must land its
# artifacts back in the source tree so the redirected ``import neml2`` (and a
# downstream C++ consumer pointed at the checkout) can find them. So for editable
# we install with ABSOLUTE destinations under ``<source>/neml2/{lib,include,share}``
# -- the same layout the shipped wheel lays down under ``<site-packages>/neml2/``,
# just rooted in the working copy. A non-editable (wheel) build uses destinations
# relative to scikit-build's install prefix, which it maps under ``neml2/``.
if(NEML2_WHEEL AND DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable")
      set(NEML2_EDITABLE ON)
else()
      set(NEML2_EDITABLE OFF)
endif()

if(NEML2_EDITABLE)
      set(INSTALL_LIBDIR ${NEML2_SOURCE_DIR}/neml2/lib)
      set(INSTALL_BINDIR ${NEML2_SOURCE_DIR}/neml2/bin)
      set(INSTALL_INCLUDEDIR ${NEML2_SOURCE_DIR}/neml2/include)
      set(INSTALL_SHAREDIR ${NEML2_SOURCE_DIR}/neml2/share)
      set(INSTALL_DATAROOT ${NEML2_SOURCE_DIR}/neml2)
else()
      set(INSTALL_LIBDIR lib)
      set(INSTALL_BINDIR bin)
      set(INSTALL_INCLUDEDIR include)
      set(INSTALL_SHAREDIR share)
      set(INSTALL_DATAROOT .)
endif()

# ----------------------------------------------------------------------------
# Torch
# ----------------------------------------------------------------------------
find_package(torch MODULE OPTIONAL_COMPONENTS cuda python)

if(NOT torch_FOUND)
      message(FATAL_ERROR
            "Torch not found. Install it into your environment and build without "
            "isolation so it is visible:\n"
            "  pip install torch nmhit\n"
            "  pip install -e \".[dev]\" --no-build-isolation")
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

# The find-modules neml2Config.cmake adds to CMAKE_MODULE_PATH at find_package()
# time (it does `find_dependency(torch MODULE)`). Shipped in both the wheel and an
# editable install via INSTALL_SHAREDIR so a downstream consumer can resolve torch.
install(FILES
      ${NEML2_SOURCE_DIR}/cmake/Modules/Findtorch.cmake
      ${NEML2_SOURCE_DIR}/cmake/Modules/DetectTorchCXXABI.cpp
      ${NEML2_SOURCE_DIR}/cmake/Modules/Findnmhit.cmake
      DESTINATION ${INSTALL_SHAREDIR}/cmake/neml2/Modules
      COMPONENT libneml2
)

# ----------------------------------------------------------------------------
# nlohmann json
# ----------------------------------------------------------------------------
# Sourced from the contrib/nlohmann_json-src git submodule and built+installed
# into contrib/nlohmann_json by contrib/install_nlohmann_json.cmake.in (a cmake -P
# script -- no shell needed, so it works on Windows/MSVC too). We auto-init the
# submodule on first configure to keep the dev experience close to
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
      custom_install(nlohmann_json contrib/install_nlohmann_json.cmake.in ${NEML2_SOURCE_DIR}/contrib/nlohmann_json-src ${NEML2_CONTRIB_PREFIX}/nlohmann_json-build ${nlohmann_json_INSTALL_PREFIX})
      find_package(nlohmann_json CONFIG REQUIRED PATHS ${nlohmann_json_INSTALL_PREFIX} NO_DEFAULT_PATH)
endif()
file(REAL_PATH "../../../" nlohmann_json_DIR BASE_DIRECTORY ${nlohmann_json_DIR})

# check if nlohmann json is the in-tree (submodule-built) copy
path_has_prefix(${nlohmann_json_DIR} ${NEML2_CONTRIB_PREFIX} nlohmann_json_CONTRIB)

# nlohmann json is packaged with the NEML2 installation if we built it ourselves
# (submodule path) or if this is a wheel build. Shipped in both the wheel and an
# editable install (into the source tree via INSTALL_INCLUDEDIR / INSTALL_SHAREDIR)
# so neml2Config's find_dependency(nlohmann_json) resolves against either.
if(nlohmann_json_CONTRIB OR NEML2_WHEEL)
      install(DIRECTORY ${nlohmann_json_DIR}/include/nlohmann
            DESTINATION ${INSTALL_INCLUDEDIR} COMPONENT libneml2)
endif()
install(DIRECTORY ${nlohmann_json_DIR}/share/ DESTINATION ${INSTALL_SHAREDIR} COMPONENT libneml2)
# nlohmann_json ships a Visual Studio debugger visualizer (nlohmann_json.natvis)
# at its install-prefix root, referenced by a Windows-only INSTALL_INTERFACE
# source on the imported nlohmann_json::nlohmann_json target. A Windows consumer
# linking neml2::aoti (which links nlohmann_json PUBLIC) therefore needs the file
# at neml2's prefix root, or find_package(neml2)'s generate step fails with
# "Cannot find source file: .../nlohmann_json.natvis". It is guarded out on
# non-Windows, so this only matters when bundled, but is harmless to ship anywhere.
if(EXISTS ${nlohmann_json_DIR}/nlohmann_json.natvis)
      install(FILES ${nlohmann_json_DIR}/nlohmann_json.natvis
            DESTINATION ${INSTALL_DATAROOT} COMPONENT libneml2)
endif()

# ----------------------------------------------------------------------------
# nmhit (C++ HIT parser)
# ----------------------------------------------------------------------------
# Discovered from the Python site-packages the current interpreter sees (same
# strategy as Findtorch); the nmhit wheel ships its static lib + headers under
# <site-packages>/nmhit/{lib,include}. Linked PRIVATE into aoti -- nmhit is an
# implementation detail of the C++ loader (factory.cpp parses the stub `.i`) and
# appears in no shipped header, so it stays out of the public ABI.
find_package(nmhit MODULE REQUIRED)

# ----------------------------------------------------------------------------
# libneml2 — the only C++ artifact NEML2 ships (aoti runtime + dispatchers)
# ----------------------------------------------------------------------------
# Free-standing: links only against torch::core (never torch::cuda -- see the
# target_link_libraries below) and nlohmann_json. Wraps
# torch::inductor::AOTIModelPackageLoader so the
# Python side can execute .pt2 artifacts produced by torch._inductor.aoti_compile_and_package
# from C++. Source tree lives at neml2/csrc/aoti/, intentionally co-located with
# the Python package so the C++ and Python sides of NEML2 stay in one tree.
# The single Model class is split across four translation units (construction,
# public ops, the value/Newton path, the Jacobian/IFT path); the internals live
# behind a PImpl `Model::Impl` declared in the non-shipped internal.h.
add_library(aoti SHARED
      neml2/csrc/aoti/Exception.cpp
      neml2/csrc/aoti/Model.cpp
      neml2/csrc/aoti/ops.cpp
      neml2/csrc/aoti/solve.cpp
      neml2/csrc/aoti/jacobian.cpp
      neml2/csrc/aoti/newton.cpp
      neml2/csrc/aoti/nonlinear_system_aoti.cpp
      neml2/csrc/aoti/custom_ops.cpp
      neml2/csrc/dispatchers/WorkScheduler.cpp
      neml2/csrc/dispatchers/SimpleScheduler.cpp
      neml2/csrc/dispatchers/MPISimpleScheduler.cpp
      neml2/csrc/dispatchers/AsyncScheduler.cpp
      neml2/csrc/dispatchers/StaticHybridScheduler.cpp
      neml2/csrc/dispatchers/DispatchedModel.cpp
      neml2/csrc/dispatchers/factory.cpp
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
      ${NEML2_SOURCE_DIR}/neml2/csrc/aoti/Exception.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/aoti/Model.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/dispatchers/WorkScheduler.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/dispatchers/SimpleScheduler.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/dispatchers/MPISimpleScheduler.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/dispatchers/AsyncScheduler.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/dispatchers/StaticHybridScheduler.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/dispatchers/DispatchedModel.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/dispatchers/factory.h
)
# The generated aoti_export.h lives under the build tree's dedicated include
# root. Add it as a build-interface include so `#include
# "neml2/csrc/aoti/aoti_export.h"` resolves while compiling the lib + the pyaoti
# binding (which links aoti and inherits this). On install it is shipped beside
# Model.h (see the install() below), and the install-interface `include` dir
# already on the target covers it.
target_include_directories(aoti PUBLIC $<BUILD_INTERFACE:${NEML2_BINARY_DIR}/include>)
set_target_properties(aoti PROPERTIES OUTPUT_NAME "neml2$<IF:$<CONFIG:Release>,,_$<CONFIG>>")
# Warning flags are compiler-specific; the GCC/Clang -W family is not understood
# by MSVC (cl.exe). /wd4251 and /wd4275 silence the "needs dll-interface" noise
# MSVC emits for the std:: members carried by value across the exported PImpl
# facade (harmless -- the DLL and its consumers use the same toolchain/runtime).
target_compile_options(aoti PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-pedantic>
      $<$<CXX_COMPILER_ID:MSVC>:/W3;/wd4251;/wd4275>)
# NOMINMAX / WIN32_LEAN_AND_MEAN: torch's headers pull in <windows.h> on MSVC,
# whose min/max macros otherwise clobber std::min/std::max (and torch's own uses).
target_compile_definitions(aoti PRIVATE
      $<$<PLATFORM_ID:Windows>:NOMINMAX;WIN32_LEAN_AND_MEAN>)
target_link_libraries(aoti PUBLIC torch::core nlohmann_json::nlohmann_json)
target_link_libraries(aoti PRIVATE nmhit::nmhit)
# Propagate the C++17 requirement to consumers via the exported target. The
# global CMAKE_CXX_STANDARD only governs THIS build; without an INTERFACE compile
# feature a downstream find_package(neml2) consumer compiles at the compiler
# default (C++14 on MSVC), which fails on the nested-namespace in Exception.h.
target_compile_features(aoti PUBLIC cxx_std_17)
# The async dispatch pool (StaticHybridScheduler path) spawns std::thread workers.
find_package(Threads REQUIRED)
target_link_libraries(aoti PRIVATE Threads::Threads)
# Deliberately NOT linked against torch::cuda. neml2's C++ references zero cuda
# symbols -- it reaches the GPU only through torch's device-generic dispatch,
# which loads libtorch_cuda at runtime when a cuda tensor appears. Linking
# torch::cuda only manufactures a hard DT_NEEDED on libc10_cuda.so that buys
# nothing and makes `import neml2` fail for anyone on a cpu-only torch (the
# library was previously linked whenever built against a cuda torch, so the
# published wheel -- built against the default-index cuda torch -- could not
# load under a cpu torch). The cpu-side torch ABI is identical across the cuda
# and cpu builds, so one cuda-free libneml2 runs under both.

# MPI scheduler: link MPI and define NEML2_MPI for the aoti TUs only. The public
# headers are MPI-type-free, so consumers need neither the flag nor mpi.h; the
# MPISimpleScheduler constructor throws when the flag is OFF.
if(NEML2_MPI)
      find_package(MPI REQUIRED COMPONENTS CXX)
      target_link_libraries(aoti PUBLIC MPI::MPI_CXX)
      target_compile_definitions(aoti PRIVATE NEML2_MPI)
      message(STATUS "NEML2_MPI=ON: MPISimpleScheduler enabled (linking ${MPI_CXX_COMPILER})")
endif()

# rpath: only the torch hop is needed (no sibling neml2_*.so libraries to
# resolve anymore).
if(NEML2_WHEEL AND NOT SKBUILD_STATE STREQUAL "editable")
      set_target_properties(aoti PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/../../torch/lib")
else()
      set_target_properties(aoti PROPERTIES INSTALL_RPATH "${torch_LINK_DIR}")
endif()

# libneml2 + its public headers + the CMake export set, in ONE install() so the
# exported target records the right LIBRARY location. Shipped in BOTH the wheel and
# an editable install (the latter into the source tree via INSTALL_LIBDIR /
# INSTALL_INCLUDEDIR) so a downstream C++ consumer can build against either: the
# wheel export is relocatable (prefix-relative); the editable export pins the
# source-tree paths. The generated export header sits beside Model.h.
install(TARGETS aoti
      EXPORT neml2targets
      # RUNTIME (the .dll) lands beside the Unix .so in lib/, matching torch's own
      # Windows layout (torch/lib/*.dll) so the DLL-search story is a single dir;
      # ARCHIVE is the import .lib a downstream C++ consumer links against.
      RUNTIME DESTINATION ${INSTALL_LIBDIR}
      LIBRARY DESTINATION ${INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${INSTALL_LIBDIR}
      FILE_SET HEADERS DESTINATION ${INSTALL_INCLUDEDIR}
      COMPONENT libneml2
)
install(FILES ${NEML2_BINARY_DIR}/include/neml2/csrc/aoti/aoti_export.h
      DESTINATION ${INSTALL_INCLUDEDIR}/neml2/csrc/aoti
      COMPONENT libneml2
)

# ----------------------------------------------------------------------------
# libneml2_eager — the embedded-Python eager runtime (separate from libneml2)
# ----------------------------------------------------------------------------
# A second shared library that embeds a CPython interpreter and runs a NEML2
# model eagerly from C++ via `neml2.factory.load_model` -- skipping the AOTI
# compile entirely (for fast downstream C++ unit tests, where a minutes-long
# `neml2-compile` is untenable). It is deliberately KEPT SEPARATE from
# `libneml2.so` (the `aoti` target above): that library links only torch::core
# and stays Python-free so the AOTI runtime can embed in pure-C++ hosts.
# `eager` is the one place the Python dependency lives. Built for the C++ tests
# and shipped in the wheel beside libneml2.so, so the add_library is top-level
# rather than gated on NEML2_WHEEL.
#
# Link against Development.Module (headers + module link flags), NOT
# Development.Embed (libpython). A shared library may leave the CPython API
# symbols undefined and resolve them at load time from whatever libpython the
# host process already has -- the Python interpreter when loaded under Python, or
# the libpython a pure-C++ host links to embed the interpreter. This mirrors the
# pybind extension modules and, crucially, removes any build-time dependency on a
# *shared* libpython, which the manylinux wheel images do not reliably provide.
#
# The eager runtime is NOT supported on Windows: leaving the CPython/torch_python
# symbols undefined for load-time resolution is a Unix (ELF/Mach-O) linking model;
# the PE/COFF linker requires every symbol in a DLL to be resolved at link time.
# On Windows we define a no-op `eager` target so scikit-build's build.targets
# (which lists "eager") still resolves; the AOTI runtime -- the C++ deployment
# path -- is unaffected. See doc: Windows is plain-AOTI only, no cpp-eager.
if(WIN32)
      add_custom_target(eager COMMENT "libneml2_eager is not built on Windows (unsupported)")
else()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# `eager` is an unconditional target, so it links `torch::python` (libtorch_python
# -> the pybind tensor caster) on every configure. That is intentional and does
# NOT compromise libneml2.so's Python-free promise -- the Python dependency lives
# only in this separate libneml2_eager. But it does require a Python-enabled
# torch: a bare C++ libtorch with no libtorch_python cannot build the eager
# runtime. Fail here with a clear message instead of a cryptic "target
# torch::python not found" deep in the link step.
if(NOT TARGET torch::python)
      message(FATAL_ERROR
            "libneml2_eager (the embedded-Python eager runtime) requires a "
            "Python-enabled torch providing libtorch_python, but torch::python "
            "was not found. The pip `torch` wheel ships it; a bare C++ libtorch "
            "does not. Install a Python torch, or drop the `eager` target (it is "
            "independent of the Python-free libneml2.so).")
endif()

add_library(eager SHARED
      neml2/csrc/eager/Model.cpp
      neml2/csrc/eager/load_model.cpp
      neml2/csrc/eager/interpreter.cpp
)

# Dedicated EAGER_EXPORT visibility macro (mirrors the aoti pattern); emit into
# the same dedicated generated-include root.
generate_export_header(eager
      BASE_NAME eager
      EXPORT_FILE_NAME ${NEML2_BINARY_DIR}/include/neml2/csrc/eager/eager_export.h
)
set_target_properties(eager PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
)
# Ship only the public, Python-free headers (Model.h + load_model.h). The
# internal.h / interpreter.h (pybind-carrying) stay out of the install.
target_sources(eager
      PUBLIC
      FILE_SET HEADERS
      BASE_DIRS ${NEML2_SOURCE_DIR}
      FILES
      ${NEML2_SOURCE_DIR}/neml2/csrc/eager/Model.h
      ${NEML2_SOURCE_DIR}/neml2/csrc/eager/load_model.h
)
target_include_directories(eager PUBLIC $<BUILD_INTERFACE:${NEML2_BINARY_DIR}/include>)
target_include_directories(eager PRIVATE ${Python3_INCLUDE_DIRS})
set_target_properties(eager PROPERTIES OUTPUT_NAME "neml2_eager$<IF:$<CONFIG:Release>,,_$<CONFIG>>")
target_compile_options(eager PRIVATE -Wall -Wextra -pedantic)
# PUBLIC: aoti (reuse the Exception ABI + headers) + torch::core. PRIVATE:
# torch::python (libtorch_python -> the pybind tensor caster) + Python3::Module
# (CPython headers; on macOS also `-undefined dynamic_lookup`). The CPython API
# symbols are left undefined and resolved from the host's libpython at load time.
# pybind11 headers come transitively via torch.
target_link_libraries(eager PUBLIC aoti torch::core)
target_link_libraries(eager PRIVATE torch::python Python3::Module)
target_compile_definitions(eager PRIVATE "PYBIND11_DETAILED_ERROR_MESSAGES")
# No torch::cuda here either, for the same reason as the aoti target above: zero
# cuda symbol references, so linking it would only add an unsatisfiable
# libc10_cuda.so DT_NEEDED on cpu-only torch.

# rpath:
#  - ${INSTALL_REL_PATH} (the lib's OWN dir): libneml2_eager.so links the sibling
#    libneml2.so (the `aoti` target), which sits next to it in neml2/lib. Without
#    this hop the loader -- and auditwheel during wheel repair -- cannot locate
#    libneml2.so ("Cannot repair wheel, because required library libneml2.so
#    could not be located").
#  - the torch hop (libtorch + libtorch_python).
# libpython is NOT linked into this lib (see Python3::Module above) -- the host
# process supplies it -- so there is nothing to point an rpath at for it.
if(NEML2_WHEEL AND NOT SKBUILD_STATE STREQUAL "editable")
      set_target_properties(eager PROPERTIES
            INSTALL_RPATH "${INSTALL_REL_PATH};${INSTALL_REL_PATH}/../../torch/lib")
else()
      set_target_properties(eager PROPERTIES
            INSTALL_RPATH "${INSTALL_REL_PATH};${torch_LINK_DIR}")
endif()

install(TARGETS eager
      EXPORT neml2targets
      LIBRARY DESTINATION ${INSTALL_LIBDIR}
      FILE_SET HEADERS DESTINATION ${INSTALL_INCLUDEDIR}
      COMPONENT libneml2
)
install(FILES ${NEML2_BINARY_DIR}/include/neml2/csrc/eager/eager_export.h
      DESTINATION ${INSTALL_INCLUDEDIR}/neml2/csrc/eager
      COMPONENT libneml2
)

endif() # NOT WIN32 -- the embedded-Python eager runtime (libneml2_eager)

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
            DESTINATION ${INSTALL_DATAROOT}
            COMPONENT libneml2
      )
endif()

# ----------------------------------------------------------------------------
# CMake package export (for downstream `find_package(neml2)`)
# ----------------------------------------------------------------------------
# Shipped in both the wheel and an editable install. The generated config is
# relocatable -- @PACKAGE_INIT@ derives PACKAGE_PREFIX_DIR from the config file's
# own location at find_package() time -- so INSTALL_DESTINATION stays the canonical
# relative `share/cmake/neml2`; only the install() destination differs (source tree
# for editable, prefix-relative for the wheel) via INSTALL_SHAREDIR.
install(EXPORT neml2targets NAMESPACE neml2:: DESTINATION ${INSTALL_SHAREDIR}/cmake/neml2 COMPONENT libneml2)

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

install(
      FILES
      ${NEML2_BINARY_DIR}/neml2Config.cmake
      ${NEML2_BINARY_DIR}/neml2ConfigVersion.cmake
      DESTINATION ${INSTALL_SHAREDIR}/cmake/neml2
      COMPONENT libneml2
)

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
            # Python extension suffix. On Windows use a plain ".pyd": CPython
            # always accepts it, and Python3_SOABI is empty there under
            # FindPython's Development.Module -- interpolating it would yield the
            # unimportable name "_aoti..pyd" (double dot). Elsewhere keep the
            # SOABI tag + platform suffix (.cpython-3XX-....so / .dylib).
            if(WIN32)
                  set(_ext_suffix ".pyd")
            else()
                  set(_ext_suffix ".${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
            endif()
            set_target_properties(${target} PROPERTIES
                  LIBRARY_OUTPUT_DIRECTORY "${_outdir}"
                  OUTPUT_NAME "_${_basename}"
                  PREFIX ""
                  SUFFIX "${_ext_suffix}"
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
      # The eager-path bridge uses pybind/py::object, so it lives in the module
      # (which links Python), not the free-standing aoti library. It drives the
      # exported C++ Newton/NonlinearSystem from the lib.
      target_sources(pyaoti PRIVATE
            ${NEML2_SOURCE_DIR}/neml2/csrc/aoti/nonlinear_system_eager.cpp
      )
endif()

# ----------------------------------------------------------------------------
# C++ tests + benchmark suite (ctest entries)
# ----------------------------------------------------------------------------
# Built for an editable install and a plain `cmake -S .` dev configure, but NOT
# baked into the shipped wheel (a shippable wheel is `NEML2_WHEEL AND NOT
# editable`). Gated additionally on CMake's canonical BUILD_TESTING so
# `-DBUILD_TESTING=OFF` turns them off as users expect. The editable scikit-build
# override adds the aggregate `cpp_tests` target to build.targets so a single
# `pip install -e` compiles them; `ctest --test-dir build/editable` then runs them.
if(BUILD_TESTING AND NOT (NEML2_WHEEL AND NOT NEML2_EDITABLE))
      add_subdirectory(benchmark)
      add_subdirectory(tests/cpp)

      # Aggregate target so `build.targets` (and humans) can build every C++ test
      # executable with one name instead of enumerating them. Collected from each
      # subdirectory's BUILDSYSTEM_TARGETS so new tests are picked up automatically.
      add_custom_target(cpp_tests)
      foreach(_dir ${NEML2_SOURCE_DIR}/benchmark ${NEML2_SOURCE_DIR}/tests/cpp)
            get_property(_dir_targets DIRECTORY ${_dir} PROPERTY BUILDSYSTEM_TARGETS)
            foreach(_t IN LISTS _dir_targets)
                  get_target_property(_t_type ${_t} TYPE)
                  if(_t_type STREQUAL "EXECUTABLE")
                        add_dependencies(cpp_tests ${_t})
                  endif()
            endforeach()
      endforeach()
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

# ----------------------------------------------------------------------------
# Stable build-dir symlink for editable installs
# ----------------------------------------------------------------------------
# scikit-build builds an editable install into build/<wheel_tag>, whose name
# varies by interpreter/platform. Drop a stable `build/editable` symlink so the
# C++ tests run from a fixed path with no preset:
#   ctest --test-dir build/editable -L dispatcher
if(NEML2_EDITABLE)
      file(CREATE_LINK ${NEML2_BINARY_DIR} ${NEML2_SOURCE_DIR}/build/editable SYMBOLIC)
endif()
