# This module is responsible for finding the WASP library
#
# Output variables:
# - WASP_FOUND
#
# This module defines the following imported targets:
# - WASP

# -----------------------------------------------------------------------------
# include directories
# -----------------------------------------------------------------------------
find_path(WASP_INCLUDE_DIR waspcore NO_CACHE)

# -----------------------------------------------------------------------------
# libraries
# -----------------------------------------------------------------------------
find_library(WASP_CORE_LIBRARY NAMES waspcore NO_CACHE)
find_library(WASP_HIT_LIBRARY NAMES wasphit NO_CACHE)

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  WASP
  REQUIRED_VARS
  WASP_INCLUDE_DIR
  WASP_CORE_LIBRARY
  WASP_HIT_LIBRARY
)

if(WASP_FOUND AND NOT TARGET WASP)
  # Figure out link directories
  # This is apparently assuming all libraries are in the same directory
  get_filename_component(WASP_LINK_DIR ${WASP_CORE_LIBRARY} DIRECTORY)

  add_library(WASP INTERFACE IMPORTED GLOBAL)
  target_include_directories(WASP INTERFACE ${WASP_INCLUDE_DIR})
  target_link_directories(WASP INTERFACE ${WASP_LINK_DIR})
  list(APPEND WASP_LIBRARIES ${WASP_CORE_LIBRARY} ${WASP_HIT_LIBRARY})
  target_link_libraries(WASP INTERFACE ${WASP_LIBRARIES})
endif()
