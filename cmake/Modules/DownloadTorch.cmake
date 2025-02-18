# Download a precompiled cpu-only libtorch.
#
# This method only works on Unix systems. The version of the downloaded libtorch
# is controlled by the TORCH_VERSION cache entry.
#
# Influential variables:
# - TORCH_VERSION
#
# Output variables are those defined by FetchContent such as:
# - torch_POPULATED
# - torch_SOURCE_DIR

if(UNIX)
  if(NOT APPLE)
    set(Torch_URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip)
  else()
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
      set(Torch_URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${TORCH_VERSION}.zip)
    else()
      set(Torch_URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${TORCH_VERSION}.zip)
    endif()
  endif()
endif()

if(DEFINED Torch_URL)
  include(FetchContent)
  FetchContent_Declare(
    torch
    URL ${Torch_URL}
  )

  message(STATUS "Downloading/updating Torch")
  FetchContent_MakeAvailable(torch)
endif()
