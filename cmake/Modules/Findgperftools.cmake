# -----------------------------------------------------------------------------
# Only define once, even if called multiple times
# -----------------------------------------------------------------------------
if(NOT DEFINED _gperftools_ALREADY_INCLUDED)
  set(_gperftools_ALREADY_INCLUDED TRUE)

  # List of valid components
  set(_gperftools_known_components profiler)
endif()

# -----------------------------------------------------------------------------
# Validate component requests
# -----------------------------------------------------------------------------
foreach(comp IN LISTS gperftools_FIND_COMPONENTS)
  list(FIND _gperftools_known_components "${comp}" _index)

  if(_index EQUAL -1)
    set(gperftools_FOUND FALSE)
    set(gperftools_NOT_FOUND_MESSAGE "Unsupported component '${comp}' requested")
    return()
  endif()
endforeach()

# -----------------------------------------------------------------------------
# Additional hints
# -----------------------------------------------------------------------------
set(_gperftools_search_paths
  ${gperftools_ROOT}
  ${GPERFTOOLS_ROOT}
  $ENV{gperftools_ROOT}
  $ENV{GPERFTOOLS_ROOT}
  $ENV{gperftools_DIR}
  $ENV{GPERFTOOLS_DIR}
)

if(NEML2_CONTRIB_PREFIX)
  list(APPEND _gperftools_search_paths ${NEML2_CONTRIB_PREFIX}/gperftools)
endif()

# -----------------------------------------------------------------------------
# gperftools::profiler
# -----------------------------------------------------------------------------
if("profiler" IN_LIST gperftools_FIND_COMPONENTS AND NOT TARGET gperftools::profiler)
  find_library(gperftools_profiler_LIBRARY NAMES profiler PATH_SUFFIXES lib HINTS ${_gperftools_search_paths})

  if(gperftools_profiler_LIBRARY)
    add_library(gperftools::profiler SHARED IMPORTED)
    get_filename_component(gperftools_profiler_LINK_DIR ${gperftools_profiler_LIBRARY} DIRECTORY)
    target_link_directories(gperftools::profiler INTERFACE ${gperftools_profiler_LINK_DIR})
    target_link_libraries(gperftools::profiler INTERFACE profiler)
    target_link_options(gperftools::profiler INTERFACE ${CMAKE_CXX_LINK_WHAT_YOU_USE_FLAG})
    set_target_properties(gperftools::profiler PROPERTIES IMPORTED_LOCATION ${gperftools_profiler_LIBRARY})
    get_filename_component(gperftools_ROOT ${gperftools_profiler_LINK_DIR} DIRECTORY)
    set(gperftools_ROOT ${gperftools_ROOT} CACHE PATH "Root directory of the gperftools installation")
    set(gperftools_profiler_FOUND TRUE)
  else()
    set(gperftools_profiler_FOUND FALSE)
  endif()
endif()

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  gperftools
  HANDLE_COMPONENTS
)
