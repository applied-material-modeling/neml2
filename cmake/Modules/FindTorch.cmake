# This module is responsible for finding the C++ backend of pytorch
#
# Additional hints are given to find_path and find_library commands if python is
# found. This is because the python package torch has the C++ backend.
#
# This module defines the following variables:
# - Torch_FOUND
# - Torch_GLIBCXX_USE_CXX11_ABI
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

  # libTorch comes with two flavors: one with cxx11 abi, one without.
  # We cache the compile definition so that we can use it in other targets.
  set(GLIBCXX_USE_CXX11_ABI 1 CACHE INTERNAL "CXX11 ABI")
  try_compile(
    Torch_GLIBCXX_USE_CXX11_ABI
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/DetectTorchCXXABI.cxx
    CMAKE_FLAGS -I${Torch_INCLUDE_DIR}
    COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}
    LINK_OPTIONS -L${Torch_LINK_DIR}
    LINK_LIBRARIES ${Torch_LIBRARY}
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED}
  )

  if(NOT Torch_GLIBCXX_USE_CXX11_ABI)
    set(GLIBCXX_USE_CXX11_ABI 0 CACHE INTERNAL "" FORCE)
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
