# This module is responsible for finding the TIMPI library
#
# Output variables:
# - TIMPI_FOUND
#
# This module defines the following imported targets:
# - TIMPI

# TIMPI depends on MPI
find_package(MPI REQUIRED)

# -----------------------------------------------------------------------------
# include directories
# -----------------------------------------------------------------------------
find_path(TIMPI_INCLUDE_DIR timpi NO_CACHE)

# -----------------------------------------------------------------------------
# libraries
# -----------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(TIMPI_BUILD_TYPE dbg)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(TIMPI_BUILD_TYPE devel)
else()
  set(TIMPI_BUILD_TYPE opt)
endif()

find_library(TIMPI_LIBRARY NAMES timpi_${TIMPI_BUILD_TYPE} NO_CACHE)

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TIMPI
  REQUIRED_VARS
  TIMPI_INCLUDE_DIR
  TIMPI_LIBRARY
)

if(TIMPI_FOUND AND NOT TARGET TIMPI)
  # Figure out link directories
  get_filename_component(TIMPI_LINK_DIR ${TIMPI_LIBRARY} DIRECTORY)

  add_library(TIMPI INTERFACE IMPORTED GLOBAL)
  target_include_directories(TIMPI INTERFACE ${TIMPI_INCLUDE_DIR})
  target_link_directories(TIMPI INTERFACE ${TIMPI_LINK_DIR})
  target_link_libraries(TIMPI INTERFACE ${TIMPI_LIBRARY} MPI::MPI_CXX)
endif()
