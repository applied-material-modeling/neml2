# -----------------------------------------------------------------------------
# Additional hints
# -----------------------------------------------------------------------------
set(_hit_search_paths
  ${hit_ROOT}
  ${HIT_ROOT}
  $ENV{hit_ROOT}
  $ENV{HIT_ROOT}
  $ENV{hit_DIR}
  $ENV{HIT_DIR}
)

if(NEML2_CONTRIB_PREFIX)
  list(APPEND _hit_search_paths ${NEML2_CONTRIB_PREFIX}/hit)
endif()

# -----------------------------------------------------------------------------
# hit
# -----------------------------------------------------------------------------
if(NOT TARGET hit)
  find_package(wasp MODULE REQUIRED COMPONENTS core hit)
  find_path(hit_INCLUDE_DIR hit PATH_SUFFIXES include HINTS ${_hit_search_paths})
  find_library(hit_LIBRARY NAMES hit PATH_SUFFIXES lib HINTS ${_hit_search_paths})

  if(hit_INCLUDE_DIR AND hit_LIBRARY)
    add_library(hit SHARED IMPORTED)
    target_include_directories(hit INTERFACE ${hit_INCLUDE_DIR})
    get_filename_component(hit_LINK_DIR ${hit_LIBRARY} DIRECTORY)
    target_link_directories(hit INTERFACE ${hit_LINK_DIR})
    target_link_libraries(hit INTERFACE ${hit_LIBRARY} wasp::core wasp::hit)
    set_target_properties(hit PROPERTIES IMPORTED_LOCATION ${hit_LIBRARY})
    get_filename_component(hit_ROOT ${hit_LINK_DIR} DIRECTORY)
    set(hit_ROOT ${hit_ROOT} CACHE PATH "Root directory of the hit installation")
    set(hit_FOUND TRUE)
  else()
    set(hit_FOUND FALSE)
  endif()
endif()

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  hit
  REQUIRED_VARS
  hit_FOUND
)
