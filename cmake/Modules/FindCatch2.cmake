# This module is responsible for finding the Catch2 library
#
# Output variables:
# - Catch2_FOUND
#
# This module defines the following imported targets:
# - Catch2::Catch2

# -----------------------------------------------------------------------------
# include directories
# -----------------------------------------------------------------------------
find_path(CATCH2_INCLUDE_DIR catch2 NO_CACHE)

# -----------------------------------------------------------------------------
# libraries
# -----------------------------------------------------------------------------
find_library(CATCH2_LIBRARY NAMES Catch2 Catch2d NO_CACHE)

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Catch2
  REQUIRED_VARS
  CATCH2_INCLUDE_DIR
  CATCH2_LIBRARY
)

if(Catch2_FOUND AND NOT TARGET Catch2::Catch2)
  # Figure out link directories
  get_filename_component(CATCH2_LINK_DIR ${CATCH2_LIBRARY} DIRECTORY)

  add_library(Catch2::Catch2 INTERFACE IMPORTED GLOBAL)
  target_include_directories(Catch2::Catch2 INTERFACE ${CATCH2_INCLUDE_DIR})
  target_link_directories(Catch2::Catch2 INTERFACE ${CATCH2_LINK_DIR})
  target_link_libraries(Catch2::Catch2 INTERFACE ${CATCH2_LIBRARY})
endif()
