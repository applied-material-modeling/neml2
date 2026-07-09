# Install a dependency by running a custom script
function(custom_install NAME SCRIPT SOURCE_DIR BINARY_DIR INSTALL_PREFIX)
  message(STATUS "Installing ${NAME}")
  # SCRIPT is a CMake-script template (configured with @ONLY, then run via
  # `cmake -P`), NOT a shell script -- so the dependency bootstrap has no bash/sh
  # dependency and works on Windows/MSVC as well as Unix.
  configure_file(${SCRIPT} ${BINARY_DIR}/install_${NAME}.cmake @ONLY)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -P ${BINARY_DIR}/install_${NAME}.cmake
    WORKING_DIRECTORY ${BINARY_DIR}
    OUTPUT_QUIET OUTPUT_FILE configure.log
    ERROR_QUIET ERROR_FILE configure.err
    RESULT_VARIABLE retcode
  )
  if(NOT retcode EQUAL 0)
    message(FATAL_ERROR "Installation of ${NAME} failed. See ${BINARY_DIR}/configure.log and ${BINARY_DIR}/configure.err for details.")
  endif()
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
