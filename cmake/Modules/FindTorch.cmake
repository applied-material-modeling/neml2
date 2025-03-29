# This module is responsible for finding the C++ backend of pytorch
#
# Additional hints are given to find_path and find_library commands if python is
# found. This is because the python package torch has the C++ backend.
#
# This module defines the following variables:
# - Torch_FOUND
# - Torch_INCLUDE_DIR, the include directory of torch
# - Torch_CSRC_INCLUDE_DIR, the include directory for the C++ API
# - C10_LIBRARY, the c10 library of pytorch
# - Torch_LIBRARY, the main torch library
# - Torch_CPU_LIBRARY, the CPU version of the torch library
# - Torch_PYTHON_LIBRARY, the python binding of the torch library
# - C10_CUDA_LIBRARY, the CUDA version of the c10 library
# - Torch_CUDA_LIBRARY, the CUDA version of the torch library
# - Torch_CUDA_LINALG_LIBRARY, the CUDA version of the torch linalg library
#
# This module defines the following internal cache entries
# - HAS_GLIBC, whether the system has glibc
# - GLIBCXX_USE_CXX11_ABI, whether the C++11 ABI is used by torch
#
# This module defines the following imported targets:
# - Torch::Torch, the C++ backend of pytorch
# - Torch::PythonBinding, the python binding of pytorch

# -----------------------------------------------------------------------------
# Hints to the find_path and find_library commands
# -----------------------------------------------------------------------------
find_package(Python3)

if(Python3_FOUND)
  set(Torch_PYTHON_PACKAGE ${Python3_SITEARCH}/torch)
  list(APPEND Torch_INCLUDE_DIR_HINTS ${Torch_PYTHON_PACKAGE}/include)
  list(APPEND Torch_LINK_DIR_HINTS ${Torch_PYTHON_PACKAGE}/lib)
endif()

# -----------------------------------------------------------------------------
# include directories
# -----------------------------------------------------------------------------
find_path(Torch_INCLUDE_DIR torch HINTS ${Torch_INCLUDE_DIR_HINTS} NO_CACHE)
find_path(Torch_CSRC_INCLUDE_DIR torch/torch.h PATHS ${Torch_INCLUDE_DIR}/torch/csrc/api/include NO_DEFAULT_PATH NO_CACHE)

# -----------------------------------------------------------------------------
# native libraries
# -----------------------------------------------------------------------------
find_library(C10_LIBRARY NAMES c10 HINTS ${Torch_LINK_DIR_HINTS} NO_CACHE)
find_library(Torch_LIBRARY NAMES torch HINTS ${Torch_LINK_DIR_HINTS} NO_CACHE)
find_library(Torch_CPU_LIBRARY NAMES torch_cpu HINTS ${Torch_LINK_DIR_HINTS} NO_CACHE)
find_library(Torch_PYTHON_LIBRARY NAMES torch_python HINTS ${Torch_LINK_DIR_HINTS} NO_CACHE)

# -----------------------------------------------------------------------------
# CUDA libraries
# -----------------------------------------------------------------------------
find_library(C10_CUDA_LIBRARY NAMES c10_cuda PATHS ${Torch_LINK_DIR} NO_CACHE)
find_library(Torch_CUDA_LIBRARY NAMES torch_cuda PATHS ${Torch_LINK_DIR} NO_CACHE)
find_library(Torch_CUDA_LINALG_LIBRARY NAMES torch_cuda_linalg PATHS ${Torch_LINK_DIR} NO_CACHE)

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Torch
  REQUIRED_VARS
  Torch_INCLUDE_DIR
  Torch_CSRC_INCLUDE_DIR
  C10_LIBRARY
  Torch_LIBRARY
  Torch_CPU_LIBRARY
  Torch_PYTHON_LIBRARY
)

if(Torch_FOUND)
  # Figure out link directories
  # This is apparently assuming all libraries are in the same directory
  get_filename_component(Torch_LINK_DIR ${Torch_LIBRARY} DIRECTORY)

  # Check if torch is a python package
  cmake_path(COMPARE ${Torch_PYTHON_PACKAGE}/lib EQUAL ${Torch_LINK_DIR} Torch_IS_PYTHON_PACKAGE)

  # Check if the system is using glibc
  if(NOT DEFINED HAS_GLIBC)
    include(CheckSymbolExists)
    check_symbol_exists(__GLIBC__ "features.h" HAS_GLIBC)
    set(HAS_GLIBC ${HAS_GLIBC} CACHE INTERNAL "System has glibc" FORCE)
  endif()

  # Detect the C++ ABI if not specified
  # This is important because the torch library may be built with either the
  # pre C++11 ABI or the C++11 ABI. We need to be consistent with the same ABI.
  if(HAS_GLIBC)
    if(DEFINED FORCE_GLIBCXX_USE_CXX11_ABI)
      set(GLIBCXX_USE_CXX11_ABI ${FORCE_GLIBCXX_USE_CXX11_ABI} CACHE INTERNAL "CXX11 ABI" FORCE)
    else()
      try_compile(
        Torch_GLIBCXX_USE_CXX11_ABI
        SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/DetectTorchCXXABI.cxx
        CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${Torch_INCLUDE_DIR}
        COMPILE_DEFINITIONS -D_GLIBCXX_USE_CXX11_ABI=1
        LINK_OPTIONS -L${Torch_LINK_DIR}
        LINK_LIBRARIES ${C10_LIBRARY}
        CXX_STANDARD ${CMAKE_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED}
      )
      try_compile(
        Torch_GLIBCXX_USE_PRE_CXX11_ABI
        SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/DetectTorchCXXABI.cxx
        CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${Torch_INCLUDE_DIR}
        COMPILE_DEFINITIONS -D_GLIBCXX_USE_CXX11_ABI=0
        LINK_OPTIONS -L${Torch_LINK_DIR}
        LINK_LIBRARIES ${C10_LIBRARY}
        CXX_STANDARD ${CMAKE_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED}
      )

      # One and only one of the two should succeed
      if(Torch_GLIBCXX_USE_CXX11_ABI STREQUAL Torch_GLIBCXX_USE_PRE_CXX11_ABI)
        message(FATAL_ERROR "Failed to detect the C++ ABI of torch. Please specify the CXX11 ABI manually using FORCE_GLIBCXX_USE_CXX11_ABI=(0|1).")
      endif()

      # Set the CXX11 ABI flag based on the detection
      if(Torch_GLIBCXX_USE_PRE_CXX11_ABI)
        set(GLIBCXX_USE_CXX11_ABI 0 CACHE INTERNAL "CXX11 ABI" FORCE)
      elseif(Torch_GLIBCXX_USE_CXX11_ABI)
        set(GLIBCXX_USE_CXX11_ABI 1 CACHE INTERNAL "CXX11 ABI" FORCE)
      endif()
    endif()
  endif()

  if(NOT TARGET Torch::Torch)
    add_library(Torch::Torch INTERFACE IMPORTED GLOBAL)
    target_link_options(Torch::Torch INTERFACE ${CMAKE_CXX_LINK_WHAT_YOU_USE_FLAG})
    target_include_directories(Torch::Torch INTERFACE ${Torch_INCLUDE_DIR} ${Torch_CSRC_INCLUDE_DIR})
    target_link_directories(Torch::Torch INTERFACE ${Torch_LINK_DIR})

    list(APPEND Torch_LIBRARIES ${C10_LIBRARY} ${Torch_LIBRARY} ${Torch_CPU_LIBRARY})

    if(Torch_C10_CUDA_LIBRARY)
      list(APPEND Torch_LIBRARIES ${C10_CUDA_LIBRARY})
    endif()

    if(Torch_CUDA_LIBRARY)
      list(APPEND Torch_LIBRARIES ${Torch_CUDA_LIBRARY})
    endif()

    if(Torch_CUDA_LINALG_LIBRARY)
      list(APPEND Torch_LIBRARIES ${Torch_CUDA_LINALG_LIBRARY})
    endif()

    target_link_libraries(Torch::Torch INTERFACE ${Torch_LIBRARIES})
  endif()

  if(NOT TARGET Torch::PythonBinding)
    add_library(Torch::PythonBinding INTERFACE IMPORTED)
    target_link_libraries(Torch::PythonBinding INTERFACE Torch::Torch ${Torch_PYTHON_LIBRARY})
  endif()
endif()
