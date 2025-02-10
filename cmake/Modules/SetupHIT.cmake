# Setup HIT
#
# This module takes care of downloading, configuring, building, and installing HIT.
# A CMakeLists.txt must exist under ${CURRENT_SOURCE_DIR}/hit
#
# Influential variables:
#
# CMAKE_BUILD_TYPE: build type. When set to "Debug", HIT will be built with
# "Debug". Otherwise, it will be built with "RelWithDebInfo".
# HIT_VERSION: version of HIT to install
# HIT_WORKING_DIR: working directory for HIT. Default is ${CMAKE_CURRENT_BINARY_DIR}/hit
# HIT_INSTALL_DIR: install directory for HIT. Default is ${FETCHCONTENT_BASE_DIR}/hit
# WASP_INCLUDE_DIRS: Include directory for WASP
# WASP_LIBRARIES: Libraries for WASP
# WASP_LINK_DIRS: Link directory for WASP libraries
# GLIBCXX_USE_CXX11_ABI: whether to use C++11 ABI. Default is 1.
#
# Exported variables:
#
# HIT_FOUND: True if HIT was successfully installed
# HIT_INCLUDE_DIRS: Include directories for HIT
# HIT_LIBRARIES: Libraries for HIT
# HIT_LINK_DIRS: Link directories for HIT libraries
#
# Exported targets:
#
# HIT: An interface library that includes HIT include directories and links HIT libraries

if(NOT HIT_FOUND)
  # build type
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(HIT_BUILD_TYPE "Debug")
  else()
    set(HIT_BUILD_TYPE "RelWithDebInfo")
  endif()

  # version
  if(NOT DEFINED HIT_VERSION)
    message(FATAL_ERROR "HIT version is not defined")
  endif()

  # working directory
  if(NOT DEFINED HIT_WORKING_DIR)
    set(HIT_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/hit)
  endif()

  # install directory
  if(NOT DEFINED HIT_INSTALL_DIR)
    set(HIT_INSTALL_DIR ${FETCHCONTENT_BASE_DIR}/hit)
  endif()

  if(NOT DEFINED WASP_INSTALL_DIR)
    set(WASP_INSTALL_DIR ${FETCHCONTENT_BASE_DIR}/WASP)
  endif()

  message(STATUS "Configuring and installing HIT (${HIT_BUILD_TYPE})")
  set(HIT_WORKING_DIR ${NEML2_BINARY_DIR}/hit)
  set(HIT_INSTALL_DIR ${FETCHCONTENT_BASE_DIR}/hit)
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${HIT_WORKING_DIR})
  execute_process(
    COMMAND ${CMAKE_COMMAND} \\
    -DVERSION=${HIT_VERSION} \\
    -DWASP_INCLUDE_DIRS=${WASP_INCLUDE_DIRS} \\
    -DWASP_LIBRARIES=${WASP_LIBRARIES} \\
    -DWASP_LINK_DIRS=${WASP_LINK_DIRS} \\
    -DBASE_DIR=${HIT_INSTALL_DIR} \\
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \\
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \\
    -DBUILD_TYPE=${HIT_BUILD_TYPE} \\
    -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \\
    -G${CMAKE_GENERATOR} \\
    ${CMAKE_CURRENT_SOURCE_DIR}/hit
    WORKING_DIRECTORY ${HIT_WORKING_DIR}
    OUTPUT_QUIET OUTPUT_FILE ${HIT_WORKING_DIR}/configure.log
    ERROR_QUIET ERROR_FILE ${HIT_WORKING_DIR}/configure.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${HIT_WORKING_DIR}
    OUTPUT_QUIET OUTPUT_FILE ${HIT_WORKING_DIR}/build.log
    ERROR_QUIET ERROR_FILE ${HIT_WORKING_DIR}/build.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --install ${HIT_WORKING_DIR} --prefix ${HIT_INSTALL_DIR}
    OUTPUT_QUIET OUTPUT_FILE ${HIT_WORKING_DIR}/install.log
    ERROR_QUIET ERROR_FILE ${HIT_WORKING_DIR}/install.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  install(DIRECTORY
    ${HIT_INSTALL_DIR}/include/hit
    TYPE INCLUDE
  )

  # Export variables
  set(HIT_FOUND TRUE)
  set(HIT_INCLUDE_DIRS ${HIT_INSTALL_DIR}/include)
  set(HIT_LIBRARIES hit)
  set(HIT_LINK_DIRS ${HIT_INSTALL_DIR}/lib)

  # Export interface
  add_library(HIT INTERFACE)
  target_include_directories(HIT SYSTEM INTERFACE ${HIT_INCLUDE_DIRS})
  target_link_directories(HIT INTERFACE ${HIT_LINK_DIRS})
  target_link_libraries(HIT INTERFACE ${HIT_LIBRARIES})
endif()
