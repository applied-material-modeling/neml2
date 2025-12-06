# -----------------------------------------------------------------------------
# Additional hints
# -----------------------------------------------------------------------------
set(_csvparser_search_paths
  ${csvparser_ROOT}
  ${CSVPARSER_ROOT}
  $ENV{csvparser_ROOT}
  $ENV{CSVPARSER_ROOT}
  $ENV{csvparser_DIR}
  $ENV{CSVPARSER_DIR}
)

if(NEML2_CONTRIB_PREFIX)
  list(APPEND _csvparser_search_paths ${NEML2_CONTRIB_PREFIX}/csvparser)
endif()

# -----------------------------------------------------------------------------
# csv.hpp
# -----------------------------------------------------------------------------
if(NOT TARGET csv)
  find_path(csvparser_INCLUDE_DIR csvparser PATH_SUFFIXES csvparser HINTS ${_csvparser_search_paths})

  if(csvparser_INCLUDE_DIR)
    add_library(csv INTERFACE IMPORTED)
    target_include_directories(csv INTERFACE ${csvparser_INCLUDE_DIR})
    get_filename_component(csvparser_ROOT ${csvparser_INCLUDE_DIR} DIRECTORY)
    set(csvparser_ROOT ${csvparser_ROOT} CACHE PATH "Root directory of the csvparser installation")
    set(csvparser_FOUND TRUE)
  else()
    set(csvparser_FOUND FALSE)
  endif()
endif()

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  csvparser
  REQUIRED_VARS
  csvparser_INCLUDE_DIR
  csvparser_FOUND
)
