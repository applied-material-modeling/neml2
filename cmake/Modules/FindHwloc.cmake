# This module is responsible for finding the HWLOC library
#
# Output variables:
# - HWLOC_FOUND
#
# This module defines the following imported targets:
# - HWLOC

# -----------------------------------------------------------------------------
# include directories
# -----------------------------------------------------------------------------
find_path(HWLOC_INCLUDE_DIR hwloc NO_CACHE)

# -----------------------------------------------------------------------------
# libraries
# -----------------------------------------------------------------------------
find_library(HWLOC_LIBRARY NAMES hwloc NO_CACHE)

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Hwloc
  REQUIRED_VARS
  HWLOC_INCLUDE_DIR
  HWLOC_LIBRARY
)

if(HWLOC_FOUND AND NOT TARGET HWLOC)
  # Figure out link directories
  get_filename_component(HWLOC_LINK_DIR ${HWLOC_LIBRARY} DIRECTORY)

  add_library(HWLOC INTERFACE IMPORTED GLOBAL)
  target_include_directories(HWLOC INTERFACE ${HWLOC_INCLUDE_DIR})
  target_link_directories(HWLOC INTERFACE ${HWLOC_LINK_DIR})
  target_link_libraries(HWLOC INTERFACE ${HWLOC_LIBRARY})
endif()
