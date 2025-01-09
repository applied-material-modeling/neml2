# This module is responsible for finding the HIT library
#
# Output variables:
# - HIT_FOUND
#
# This module defines the following imported library:
# - HIT

find_package(WASP)

# -----------------------------------------------------------------------------
# include directories
# -----------------------------------------------------------------------------
find_path(HIT_INCLUDE_DIR hit NO_CACHE)

# -----------------------------------------------------------------------------
# libraries
# -----------------------------------------------------------------------------
find_library(HIT_LIBRARY NAMES hit NO_CACHE)

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  HIT
  REQUIRED_VARS
  WASP_FOUND
  HIT_INCLUDE_DIR
  HIT_LIBRARY
)

if(HIT_FOUND AND NOT TARGET HIT)
  # Figure out link directories
  get_filename_component(HIT_LINK_DIR ${HIT_LIBRARY} DIRECTORY)

  add_library(HIT INTERFACE IMPORTED GLOBAL)
  target_include_directories(HIT INTERFACE ${HIT_INCLUDE_DIR})
  target_link_directories(HIT INTERFACE ${HIT_LINK_DIR})
  target_link_libraries(HIT INTERFACE ${HIT_LIBRARY} WASP)
endif()
