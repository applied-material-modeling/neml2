# -----------------------------------------------------------------------------
# Findnmhit
# -----------------------------------------------------------------------------
# Locate the C++ nmhit library (the NEML2-flavored HIT parser) the same way
# Findtorch locates libtorch: from the Python site-packages the current
# interpreter sees. The nmhit wheel ships its static archive + public headers
# under <site-packages>/nmhit/{lib,include} (see neml2-hit's CMakeLists), so the
# version that `import nmhit` resolves is exactly the one we link against -- the
# reliable choice in venvs, PEP517-isolated builds, and multi-version setups.
#
# Honors nmhit_ROOT / $ENV{nmhit_ROOT} as hints for a source-tree / build-tree
# install (e.g. an editable dev checkout of neml2-hit).
#
# Defines the imported target nmhit::nmhit and sets nmhit_FOUND.

if(NOT TARGET nmhit::nmhit)
  set(_nmhit_search_paths
    ${nmhit_ROOT}
    ${NMHIT_ROOT}
    $ENV{nmhit_ROOT}
    $ENV{NMHIT_ROOT}
  )

  # nmhit as seen by the current Python interpreter.
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -m site --user-site
      OUTPUT_VARIABLE _nmhit_user_site
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(EXISTS "${_nmhit_user_site}")
      list(APPEND _nmhit_search_paths ${_nmhit_user_site}/nmhit)
    endif()
    if(EXISTS "${Python3_SITEARCH}")
      list(APPEND _nmhit_search_paths ${Python3_SITEARCH}/nmhit)
    endif()
  endif()

  find_path(nmhit_INCLUDE_DIR nmhit/nmhit.h
    PATH_SUFFIXES include
    HINTS ${_nmhit_search_paths}
    NO_DEFAULT_PATH
  )
  find_library(nmhit_LIBRARY
    NAMES nmhit
    PATH_SUFFIXES lib
    HINTS ${_nmhit_search_paths}
  )

  if(nmhit_INCLUDE_DIR AND nmhit_LIBRARY)
    get_filename_component(nmhit_LINK_DIR ${nmhit_LIBRARY} DIRECTORY)
    add_library(nmhit::nmhit INTERFACE IMPORTED)
    # SYSTEM so nmhit's headers don't trip aoti's -Wall -Wextra -pedantic.
    target_include_directories(nmhit::nmhit SYSTEM INTERFACE ${nmhit_INCLUDE_DIR})
    target_link_libraries(nmhit::nmhit INTERFACE ${nmhit_LIBRARY})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nmhit
  REQUIRED_VARS nmhit_LIBRARY nmhit_INCLUDE_DIR
)
