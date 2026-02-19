# -----------------------------------------------------------------------------
# Only define once, even if called multiple times
# -----------------------------------------------------------------------------
if(NOT DEFINED _torch_ALREADY_INCLUDED)
  set(_torch_ALREADY_INCLUDED TRUE)

  # List of valid components
  set(_torch_known_components core cuda python)
endif()

# -----------------------------------------------------------------------------
# Validate component requests
# -----------------------------------------------------------------------------
foreach(comp IN LISTS torch_FIND_COMPONENTS)
  list(FIND _torch_known_components "${comp}" _index)

  if(_index EQUAL -1)
    set(torch_FOUND FALSE)
    set(torch_NOT_FOUND_MESSAGE "Unsupported component '${comp}' requested")
    return()
  endif()
endforeach()

# -----------------------------------------------------------------------------
# Additional hints
# -----------------------------------------------------------------------------
set(_torch_search_paths
  ${torch_ROOT}
  ${TORCH_ROOT}
  $ENV{torch_ROOT}
  $ENV{TORCH_ROOT}
  $ENV{torch_DIR}
  $ENV{TORCH_DIR}
)

# PyTorch that the current python executable sees.
# This is the most reliable way to find the correct torch installation, especially
# in virtual envs, PEP517 isolated environments, and environments with multiple
# versions of torch installed.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND AND torch_SEARCH_SITE_PACKAGES)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m site --user-site
    OUTPUT_VARIABLE Python3_USER_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(EXISTS "${Python3_USER_SITE}")
    list(APPEND _torch_search_paths ${Python3_USER_SITE}/torch)
  endif()

  if(EXISTS "${Python3_SITEARCH}")
    list(APPEND _torch_search_paths ${Python3_SITEARCH}/torch)
  endif()
endif()

if(NEML2_CONTRIB_PREFIX)
  list(APPEND _torch_search_paths ${NEML2_CONTRIB_PREFIX}/torch-src)
endif()

# -----------------------------------------------------------------------------
# torch::core
# -----------------------------------------------------------------------------
if(NOT TARGET torch::core)
  find_path(torch_INCLUDE_DIR torch PATH_SUFFIXES include HINTS ${_torch_search_paths} NO_DEFAULT_PATH)
  find_path(torch_csrc_INCLUDE_DIR torch/torch.h PATHS ${torch_INCLUDE_DIR}/torch/csrc/api/include NO_DEFAULT_PATH)
  find_library(c10_LIBRARY NAMES c10 PATH_SUFFIXES lib HINTS ${_torch_search_paths})
  find_library(torch_LIBRARY NAMES torch PATH_SUFFIXES lib HINTS ${_torch_search_paths})
  find_library(torch_cpu_LIBRARY NAMES torch_cpu PATH_SUFFIXES lib HINTS ${_torch_search_paths})

  if(torch_INCLUDE_DIR AND torch_csrc_INCLUDE_DIR AND c10_LIBRARY AND torch_LIBRARY AND torch_cpu_LIBRARY)
    # Make sure the link directories are consistent
    get_filename_component(c10_LINK_DIR ${c10_LIBRARY} DIRECTORY)
    get_filename_component(torch_LINK_DIR ${torch_LIBRARY} DIRECTORY)
    get_filename_component(torch_cpu_LINK_DIR ${torch_cpu_LIBRARY} DIRECTORY)

    if(NOT c10_LINK_DIR STREQUAL torch_LINK_DIR OR NOT c10_LINK_DIR STREQUAL torch_cpu_LINK_DIR)
      set(torch_NOT_FOUND_MESSAGE "Inconsistent link directories for torch libraries. This might happen if multiple versions of torch are installed. Please ensure all torch libraries are from the same version. libc10 is found in ${c10_LINK_DIR}, libtorch in ${torch_LINK_DIR}, and libtorch_cpu in ${torch_cpu_LINK_DIR}.")
      return()
    endif()

    add_library(torch::core INTERFACE IMPORTED)
    target_include_directories(torch::core SYSTEM INTERFACE ${torch_INCLUDE_DIR} ${torch_csrc_INCLUDE_DIR})
    target_link_directories(torch::core INTERFACE ${c10_LINK_DIR})
    target_link_libraries(torch::core INTERFACE ${c10_LIBRARY} ${torch_LIBRARY} ${torch_cpu_LIBRARY})
    get_filename_component(torch_ROOT ${torch_LINK_DIR} DIRECTORY)
    set(torch_ROOT ${torch_ROOT} CACHE PATH "Root directory of the torch installation")
    set(torch_core_FOUND TRUE)

    # Check if the system is using glibc
    include(CheckSymbolExists)
    check_symbol_exists(__GLIBC__ "features.h" HAS_GLIBC)

    # Detect the C++ ABI if not specified
    # This is important because the torch library may be built with either the
    # pre C++11 ABI or the C++11 ABI. We need to be consistent with the same ABI.
    if(HAS_GLIBC)
      try_compile(
        torch_GLIBCXX_USE_CXX11_ABI
        SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/DetectTorchCXXABI.cxx
        CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${torch_INCLUDE_DIR}
        COMPILE_DEFINITIONS -D_GLIBCXX_USE_CXX11_ABI=1
        LINK_OPTIONS -L${torch_LINK_DIR}
        LINK_LIBRARIES ${c10_LIBRARY}
        CXX_STANDARD 17
      )
      try_compile(
        torch_GLIBCXX_USE_PRE_CXX11_ABI
        SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/DetectTorchCXXABI.cxx
        CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${torch_INCLUDE_DIR}
        COMPILE_DEFINITIONS -D_GLIBCXX_USE_CXX11_ABI=0
        LINK_OPTIONS -L${torch_LINK_DIR}
        LINK_LIBRARIES ${c10_LIBRARY}
        CXX_STANDARD 17
      )

      # One and only one of the two should succeed
      if(torch_GLIBCXX_USE_CXX11_ABI STREQUAL torch_GLIBCXX_USE_PRE_CXX11_ABI)
        message(FATAL_ERROR "Failed to detect the C++11 ABI of torch. Please specify the C++11 ABI manually using FORCE_GLIBCXX_USE_CXX11_ABI=(0|1).")
      endif()

      # Set the CXX11 ABI flag based on the detection
      if(torch_GLIBCXX_USE_PRE_CXX11_ABI)
        message(FATAL_ERROR "Detected that torch is built with the pre C++11 ABI which is no longer supported. Please upgrade your torch installation.")
      endif()
    endif()
  else()
    set(torch_core_FOUND FALSE)
  endif()
endif()

# -----------------------------------------------------------------------------
# torch::cuda
# -----------------------------------------------------------------------------
if("cuda" IN_LIST torch_FIND_COMPONENTS AND NOT TARGET torch::cuda)
  if(NOT TARGET torch::core)
    set(torch_NOT_FOUND_MESSAGE "torch::cuda requires torch::core, which was not found")
  else()
    find_library(c10_cuda_LIBRARY NAMES c10_cuda PATH_SUFFIXES lib HINTS ${_torch_search_paths})
    find_library(torch_cuda_LIBRARY NAMES torch_cuda PATH_SUFFIXES lib HINTS ${_torch_search_paths})
    find_library(torch_cuda_linalg_LIBRARY NAMES torch_cuda_linalg PATH_SUFFIXES lib HINTS ${_torch_search_paths})

    if(c10_cuda_LIBRARY AND torch_cuda_LIBRARY AND torch_cuda_linalg_LIBRARY)
      # Make sure the link directories are consistent
      get_filename_component(c10_cuda_LINK_DIR ${c10_cuda_LIBRARY} DIRECTORY)
      get_filename_component(torch_cuda_LINK_DIR ${torch_cuda_LIBRARY} DIRECTORY)
      get_filename_component(torch_cuda_linalg_LINK_DIR ${torch_cuda_linalg_LIBRARY} DIRECTORY)

      if(NOT c10_cuda_LINK_DIR STREQUAL torch_cuda_LINK_DIR OR NOT c10_cuda_LINK_DIR STREQUAL torch_cuda_linalg_LINK_DIR)
        set(torch_NOT_FOUND_MESSAGE "Inconsistent link directories for torch cuda libraries. This might happen if multiple versions of torch are installed. Please ensure all torch libraries are from the same version. libc10_cuda is found in ${c10_cuda_LINK_DIR}, libtorch_cuda is found in ${torch_cuda_LINK_DIR}, and libtorch_cuda_linalg is found in ${torch_cuda_linalg_LINK_DIR}.")
        return()
      endif()

      add_library(torch::cuda INTERFACE IMPORTED)
      target_link_directories(torch::cuda INTERFACE ${c10_cuda_LINK_DIR})
      target_link_libraries(torch::cuda INTERFACE torch::core ${c10_cuda_LIBRARY} ${torch_cuda_LIBRARY} ${torch_cuda_linalg_LIBRARY})
      set(torch_cuda_FOUND TRUE)
    else()
      set(torch_cuda_FOUND FALSE)
    endif()
  endif()
endif()

# -----------------------------------------------------------------------------
# torch::python
# -----------------------------------------------------------------------------
if("python" IN_LIST torch_FIND_COMPONENTS AND NOT TARGET torch::python)
  if(NOT TARGET torch::core)
    set(torch_NOT_FOUND_MESSAGE "torch::python requires torch::core, which was not found")
  else()
    find_library(torch_python_LIBRARY NAMES torch_python PATH_SUFFIXES lib HINTS ${_torch_search_paths})

    if(torch_python_LIBRARY)
      get_filename_component(torch_python_LINK_DIR ${torch_python_LIBRARY} DIRECTORY)
      add_library(torch::python INTERFACE IMPORTED)
      target_link_directories(torch::python INTERFACE ${torch_python_LINK_DIR})
      target_link_libraries(torch::python INTERFACE torch::core ${torch_python_LIBRARY})
      set(torch_python_FOUND TRUE)
    else()
      set(torch_python_FOUND FALSE)
    endif()
  endif()
endif()

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  torch
  REQUIRED_VARS
  torch_LINK_DIR
  torch_core_FOUND
  HANDLE_COMPONENTS
)
