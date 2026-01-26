# -----------------------------------------------------------------------------
# Additional hints
# -----------------------------------------------------------------------------
set(_timpi_search_paths
  ${timpi_ROOT}
  ${TIMPI_ROOT}
  $ENV{timpi_ROOT}
  $ENV{TIMPI_ROOT}
  $ENV{timpi_DIR}
  $ENV{TIMPI_DIR}
)

if(NEML2_CONTRIB_PREFIX)
  list(APPEND _timpi_search_paths ${NEML2_CONTRIB_PREFIX}/timpi)
endif()

# -----------------------------------------------------------------------------
# timpi
# -----------------------------------------------------------------------------
if(NOT TARGET timpi)
  find_package(MPI REQUIRED)
  find_path(timpi_INCLUDE_DIR timpi PATH_SUFFIXES include HINTS ${_timpi_search_paths})
  find_library(timpi_LIBRARY NAMES timpi_${timpi_BUILD_TYPE} PATH_SUFFIXES lib HINTS ${_timpi_search_paths})

  if(timpi_INCLUDE_DIR AND timpi_LIBRARY)
    add_library(timpi SHARED IMPORTED)
    target_include_directories(timpi INTERFACE ${timpi_INCLUDE_DIR})
    get_filename_component(timpi_LINK_DIR ${timpi_LIBRARY} DIRECTORY)
    target_link_directories(timpi INTERFACE ${timpi_LINK_DIR})
    target_link_libraries(timpi INTERFACE MPI::MPI_CXX ${timpi_LIBRARY})
    set_target_properties(timpi PROPERTIES IMPORTED_LOCATION ${timpi_LIBRARY})
    get_filename_component(timpi_ROOT ${timpi_LINK_DIR} DIRECTORY)
    set(timpi_ROOT ${timpi_ROOT} CACHE PATH "Root directory of the timpi installation")
    set(timpi_FOUND TRUE)
  else()
    set(timpi_FOUND FALSE)
  endif()
endif()

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  timpi
  REQUIRED_VARS
  timpi_LINK_DIR
  timpi_FOUND
)
