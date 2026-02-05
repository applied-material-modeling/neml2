# Download code from url
function(download_from_url NAME URL PREFIX)
  message(STATUS "Downloading/updating ${NAME}")
  include(FetchContent)
  set(SOURCE_DIR ${PREFIX}/${NAME}-src)
  set(SUBBUILD_DIR ${PREFIX}/${NAME}-subbuild)
  set(BINARY_DIR ${PREFIX}/${NAME}-build)
  FetchContent_Populate(
    ${NAME}
    QUIET
    URL ${URL}
    SOURCE_DIR ${SOURCE_DIR}
    BINARY_DIR ${BINARY_DIR}
    SUBBUILD_DIR ${SUBBUILD_DIR}
  )
endfunction()

# Download code from git
function(download_from_git NAME REPO PREFIX TAG)
  message(STATUS "Downloading/updating ${NAME}")
  include(FetchContent)
  set(SOURCE_DIR ${PREFIX}/${NAME}-src)
  set(SUBBUILD_DIR ${PREFIX}/${NAME}-subbuild)
  set(BINARY_DIR ${PREFIX}/${NAME}-build)
  FetchContent_Populate(
    ${NAME}
    QUIET
    GIT_REPOSITORY ${REPO}
    GIT_TAG ${TAG}
    SOURCE_DIR ${SOURCE_DIR}
    BINARY_DIR ${BINARY_DIR}
    SUBBUILD_DIR ${SUBBUILD_DIR}
  )
endfunction()

# Install a dependency by running a custom script
function(custom_install NAME SCRIPT SOURCE_DIR BINARY_DIR INSTALL_PREFIX)
  message(STATUS "Installing ${NAME}")
  configure_file(${SCRIPT} ${BINARY_DIR}/install_${NAME}.sh)
  execute_process(
    COMMAND bash -c ./install_${NAME}.sh
    WORKING_DIRECTORY ${BINARY_DIR}
    OUTPUT_QUIET OUTPUT_FILE configure.log
    ERROR_QUIET ERROR_FILE configure.err
    RESULT_VARIABLE retcode
  )
  if(NOT retcode EQUAL 0)
    message(FATAL_ERROR "Installation of ${NAME} failed. See ${BINARY_DIR}/configure.log and ${BINARY_DIR}/configure.err for details.")
  endif()
endfunction()

# Install a file and all files with the same name but different extensions
function(install_glob path install_dir component)
  get_filename_component(dir ${path} DIRECTORY)
  get_filename_component(filename ${path} NAME_WE)
  file(GLOB files "${dir}/${filename}*")

  if(NOT files)
    message(WARNING "No files found for pattern: ${pattern}")
    return()
  endif()

  install(FILES ${files} DESTINATION "${install_dir}" COMPONENT ${component})
endfunction()

# Check if a path has a specific prefix
#
# Usage:
#   path_has_prefix(<path> <prefix> <result_var>)
#   Sets <result_var> to TRUE if <path> is under <prefix>, else FALSE.
function(path_has_prefix path prefix result_var)
  # Resolve to absolute real paths (follows symlinks, normalizes)
  get_filename_component(abs_path "${path}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  get_filename_component(abs_prefix "${prefix}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

  # Add trailing slash to prefix for strict matching
  set(abs_prefix_slash "${abs_prefix}/")

  # Check either exact match or "prefix/" match
  if(abs_path STREQUAL abs_prefix OR abs_path MATCHES "^${abs_prefix_slash}")
    set(${result_var} TRUE PARENT_SCOPE)
  else()
    set(${result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Check if a Python module can be imported
#
# Usage:
#   check_python_import(<module>)
#
# module is a dotted name like "numpy" or "torch.utils.cpp_extension"
function(check_python_import module)
  # fatal error is python not found
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python executable not found. Cannot check for Python module: ${module}")
  endif()

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import importlib; importlib.import_module('${module}')"
    RESULT_VARIABLE _rc
    OUTPUT_QUIET
    ERROR_VARIABLE _err
  )

  if(NOT _rc EQUAL 0)
    string(REPLACE "\n" "\n  " _err_indented "${_err}")
    message(FATAL_ERROR
      "Python import check failed:\n"
      "  Python: ${Python3_EXECUTABLE}\n"
      "  Module: ${module}\n"
      "  Error:\n  ${_err_indented}\n"
    )
  endif()
endfunction()
