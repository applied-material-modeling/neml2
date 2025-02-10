# Setup WASP
#
# This module takes care of downloading, configuring, building, and installing WASP.
# A CMakeLists.txt must exist under ${CURRENT_SOURCE_DIR}/wasp, inside which WASP is
# treated as an external project.
#
# Influential variables:
#
# CMAKE_BUILD_TYPE: build type. When set to "Debug", WASP will be built with
# "Debug". Otherwise, it will be built with "RelWithDebInfo".
# WASP_VERSION: version of WASP to install
# WASP_WORKING_DIR: working directory for WASP. Default is ${CMAKE_CURRENT_BINARY_DIR}/wasp
# WASP_INSTALL_DIR: install directory for WASP. Default is ${FETCHCONTENT_BASE_DIR}/wasp
# GLIBCXX_USE_CXX11_ABI: whether to use C++11 ABI. Default is 1.
#
# Exported variables:
#
# WASP_FOUND: True if WASP was successfully installed
# WASP_INCLUDE_DIRS: Include directories for WASP
# WASP_LIBRARIES: Libraries for WASP
# WASP_LINK_DIRS: Link directories for WASP libraries
#
# Exported targets:
#
# WASP: An interface library that includes WASP include directories and links WASP libraries

if(NOT WASP_FOUND)
  # build type
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(WASP_BUILD_TYPE "Debug")
  else()
    set(WASP_BUILD_TYPE "RelWithDebInfo")
  endif()

  # version
  if(NOT DEFINED WASP_VERSION)
    message(FATAL_ERROR "WASP version is not defined")
  endif()

  # working directory
  if(NOT DEFINED WASP_WORKING_DIR)
    set(WASP_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/wasp)
  endif()

  # install directory
  if(NOT DEFINED WASP_INSTALL_DIR)
    set(WASP_INSTALL_DIR ${FETCHCONTENT_BASE_DIR}/wasp)
  endif()

  # cxx abi
  if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
    message(WARNING "GLIBCXX_USE_CXX11_ABI is not defined. Defaulting to 1.")
    set(GLIBCXX_USE_CXX11_ABI 1)
  endif()

  message(STATUS "Configuring and installing WASP (${WASP_BUILD_TYPE})")

  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${WASP_WORKING_DIR})
  execute_process(
    COMMAND ${CMAKE_COMMAND} \\
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \\
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \\
    -DVERSION=${WASP_VERSION} \\
    -DBASE_DIR=${WASP_INSTALL_DIR} \\
    -DBUILD_TYPE=${WASP_BUILD_TYPE} \\
    -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \\
    ${CMAKE_CURRENT_SOURCE_DIR}/wasp
    WORKING_DIRECTORY ${WASP_WORKING_DIR}
    OUTPUT_QUIET OUTPUT_FILE ${WASP_WORKING_DIR}/configure.log
    ERROR_QUIET ERROR_FILE ${WASP_WORKING_DIR}/configure.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${WASP_WORKING_DIR}
    OUTPUT_QUIET OUTPUT_FILE ${WASP_WORKING_DIR}/build.log
    ERROR_QUIET ERROR_FILE ${WASP_WORKING_DIR}/build.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  install(DIRECTORY
    ${WASP_INSTALL_DIR}/include/waspcore
    ${WASP_INSTALL_DIR}/include/wasphit
    TYPE INCLUDE
  )

  # Export variables
  set(WASP_FOUND TRUE)
  set(WASP_INCLUDE_DIRS ${WASP_INSTALL_DIR}/include)
  list(APPEND WASP_LIBRARIES waspcore wasphit)
  set(WASP_LIBRARIES ${WASP_LIBRARIES})
  set(WASP_LINK_DIRS ${WASP_INSTALL_DIR}/lib)

  # Export interface
  add_library(WASP INTERFACE)
  target_include_directories(WASP SYSTEM INTERFACE ${WASP_INCLUDE_DIRS})
  target_link_directories(WASP INTERFACE ${WASP_LINK_DIRS})
  target_link_libraries(WASP INTERFACE ${WASP_LIBRARIES})
endif()
