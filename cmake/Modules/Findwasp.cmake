# -----------------------------------------------------------------------------
# Only define once, even if called multiple times
# -----------------------------------------------------------------------------
if(NOT DEFINED _wasp_ALREADY_INCLUDED)
  set(_wasp_ALREADY_INCLUDED TRUE)

  # List of valid components
  set(_wasp_known_components core hit)
endif()

# -----------------------------------------------------------------------------
# Validate component requests
# -----------------------------------------------------------------------------
foreach(comp IN LISTS wasp_FIND_COMPONENTS)
  list(FIND _wasp_known_components "${comp}" _index)

  if(_index EQUAL -1)
    set(wasp_FOUND FALSE)
    set(wasp_NOT_FOUND_MESSAGE "Unsupported component '${comp}' requested")
    return()
  endif()
endforeach()

# -----------------------------------------------------------------------------
# Additional hints
# -----------------------------------------------------------------------------
set(_wasp_search_paths
  ${wasp_ROOT}
  ${WASP_ROOT}
  $ENV{wasp_ROOT}
  $ENV{WASP_ROOT}
  $ENV{wasp_DIR}
  $ENV{WASP_DIR}
)

if(NEML2_CONTRIB_PREFIX)
  list(APPEND _wasp_search_paths ${NEML2_CONTRIB_PREFIX}/wasp)
endif()

# -----------------------------------------------------------------------------
# wasp::core
# -----------------------------------------------------------------------------
if(NOT TARGET wasp::core)
  find_path(wasp_core_INCLUDE_DIR waspcore PATH_SUFFIXES include HINTS ${_wasp_search_paths})
  find_library(wasp_core_LIBRARY waspcore PATH_SUFFIXES lib HINTS ${_wasp_search_paths})

  if(wasp_core_INCLUDE_DIR AND wasp_core_LIBRARY)
    add_library(wasp::core SHARED IMPORTED)
    target_include_directories(wasp::core INTERFACE ${wasp_core_INCLUDE_DIR})
    get_filename_component(wasp_core_LINK_DIR ${wasp_core_LIBRARY} DIRECTORY)
    target_link_directories(wasp::core INTERFACE ${wasp_core_LINK_DIR})
    target_link_libraries(wasp::core INTERFACE ${wasp_core_LIBRARY})
    set_target_properties(wasp::core PROPERTIES IMPORTED_LOCATION ${wasp_core_LIBRARY})
    get_filename_component(wasp_ROOT ${wasp_core_LINK_DIR} DIRECTORY)
    set(wasp_ROOT ${wasp_ROOT} CACHE PATH "Root directory of the wasp installation")
    set(wasp_core_FOUND TRUE)
  else()
    set(wasp_core_FOUND FALSE)
  endif()
endif()

# -----------------------------------------------------------------------------
# wasp::hit
# -----------------------------------------------------------------------------
if("hit" IN_LIST wasp_FIND_COMPONENTS AND NOT TARGET wasp::hit)
  if(NOT TARGET wasp::core)
    set(wasp_NOT_FOUND_MESSAGE "wasp::hit requires wasp::core, which was not found")
  else()
    find_path(wasp_hit_INCLUDE_DIR wasphit PATH_SUFFIXES include HINTS ${_wasp_search_paths})
    find_library(wasp_hit_LIBRARY wasphit PATH_SUFFIXES lib HINTS ${_wasp_search_paths})

    if(wasp_hit_INCLUDE_DIR AND wasp_hit_LIBRARY)
      add_library(wasp::hit SHARED IMPORTED)
      target_link_libraries(wasp::hit INTERFACE wasp::core)
      get_filename_component(wasp_hit_LINK_DIR ${wasp_hit_LIBRARY} DIRECTORY)
      target_include_directories(wasp::hit INTERFACE ${wasp_hit_INCLUDE_DIR})
      target_link_directories(wasp::hit INTERFACE ${wasp_hit_LINK_DIR})
      target_link_libraries(wasp::hit INTERFACE ${wasp_hit_LIBRARY})
      set_target_properties(wasp::hit PROPERTIES IMPORTED_LOCATION ${wasp_hit_LIBRARY})
      set(wasp_hit_FOUND TRUE)
    else()
      set(wasp_hit_FOUND FALSE)
    endif()
  endif()
endif()

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  wasp
  REQUIRED_VARS
  wasp_core_FOUND
  HANDLE_COMPONENTS
)
