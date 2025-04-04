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
find_library(WASP_DDI_LIBRARY NAMES waspddi NO_CACHE)
find_library(WASP_EXPR_LIBRARY NAMES waspexpr NO_CACHE)
find_library(WASP_HALITE_LIBRARY NAMES wasphalite NO_CACHE)
find_library(WASP_HIT_LIBRARY NAMES wasphit NO_CACHE)
find_library(WASP_HIVE_LIBRARY NAMES wasphive NO_CACHE)
find_library(WASP_JSON_LIBRARY NAMES waspjson NO_CACHE)
find_library(WASP_LSP_LIBRARY NAMES wasplsp NO_CACHE)
find_library(WASP_SIREN_LIBRARY NAMES waspsiren NO_CACHE)
find_library(WASP_SON_LIBRARY NAMES waspson NO_CACHE)

# -----------------------------------------------------------------------------
# Check if we found everything
# -----------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  WASP
  REQUIRED_VARS
  WASP_INCLUDE_DIR
  WASP_CORE_LIBRARY
  WASP_DDI_LIBRARY
  WASP_EXPR_LIBRARY
  WASP_HALITE_LIBRARY
  WASP_HIT_LIBRARY
  WASP_HIVE_LIBRARY
  WASP_JSON_LIBRARY
  WASP_LSP_LIBRARY
  WASP_SIREN_LIBRARY
  WASP_SON_LIBRARY
)

if(WASP_FOUND AND NOT TARGET WASP)
  # Figure out link directories
  # This is apparently assuming all libraries are in the same directory
  get_filename_component(WASP_LINK_DIR ${WASP_CORE_LIBRARY} DIRECTORY)

  add_library(WASP INTERFACE IMPORTED GLOBAL)
  target_include_directories(WASP INTERFACE ${WASP_INCLUDE_DIR})
  target_link_directories(WASP INTERFACE ${WASP_LINK_DIR})
  list(APPEND WASP_LIBRARIES
    ${WASP_CORE_LIBRARY}
    ${WASP_DDI_LIBRARY}
    ${WASP_EXPR_LIBRARY}
    ${WASP_HALITE_LIBRARY}
    ${WASP_HIT_LIBRARY}
    ${WASP_HIVE_LIBRARY}
    ${WASP_LSP_LIBRARY}
    ${WASP_JSON_LIBRARY}
    ${WASP_SIREN_LIBRARY}
    ${WASP_SON_LIBRARY}
  )
  target_link_libraries(WASP INTERFACE ${WASP_LIBRARIES})
endif()
