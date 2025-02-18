# This module is responsible for finding the Gperftools library
#
# Output variables:
# - Gperftools_FOUND
#
# This module defines the following imported targets:
# - Gperftools::profiler

# -----------------------------------------------------------------------------
# include directories
# -----------------------------------------------------------------------------
find_path(Gperftools_INCLUDE_DIR gperftools NO_CACHE)

# -----------------------------------------------------------------------------
# libraries
# -----------------------------------------------------------------------------
find_library(Gperftools_PROFILER_LIBRARY NAMES profiler NO_CACHE)

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Gperftools
  REQUIRED_VARS
  Gperftools_INCLUDE_DIR
  Gperftools_PROFILER_LIBRARY
)

if(Gperftools_FOUND)
  # Figure out link directories
  get_filename_component(Gperftools_LINK_DIR ${Gperftools_PROFILER_LIBRARY} DIRECTORY)

  if(NOT TARGET Gperftools::profiler)
    add_library(Gperftools::profiler INTERFACE IMPORTED)
    target_include_directories(Gperftools::profiler INTERFACE ${Gperftools_INCLUDE_DIR})
    target_link_directories(Gperftools::profiler INTERFACE ${Gperftools_LINK_DIR})
    target_link_libraries(Gperftools::profiler INTERFACE ${Gperftools_PROFILER_LIBRARY})
  endif()
endif()
