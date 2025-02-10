# Setup TIMPI
#
# This module takes care of downloading, configuring, building, and installing TIMPI.
# A CMakeLists.txt must exist under ${CURRENT_SOURCE_DIR}/timpi
#
# Influential variables:
#
# CMAKE_BUILD_TYPE: build type. When set to "Debug", TIMPI will be built with "dbg".
# When set to "Release", TIMPI will be built with "opt". Otherwise, it will be built with "devel".
# MPI_C_COMPILER: MPI C compiler
# MPI_CXX_COMPILER: MPI C++ compiler
# TIMPI_VERSION: version of TIMPI to install
# TIMPI_WORKING_DIR: working directory for TIMPI. Default is ${CMAKE_CURRENT_BINARY_DIR}/timpi
# TIMPI_INSTALL_DIR: install directory for TIMPI. Default is ${FETCHCONTENT_BASE_DIR}/timpi
#
# Exported variables:
#
# TIMPI_FOUND: True if TIMPI was successfully installed
# TIMPI_INCLUDE_DIRS: Include directories for TIMPI
# TIMPI_LIBRARIES: Libraries for TIMPI
# TIMPI_LINK_DIRS: Link directories for TIMPI libraries

if(NOT TIMPI_FOUND)
  # build type
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TIMPI_BUILD_TYPE "dbg")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(TIMPI_BUILD_TYPE "opt")
  else()
    set(TIMPI_BUILD_TYPE "devel")
  endif()

  # version
  if(NOT DEFINED TIMPI_VERSION)
    message(FATAL_ERROR "TIMPI version is not defined")
  endif()

  # MPI compilers
  if(NOT DEFINED MPI_C_COMPILER)
    message(FATAL_ERROR "MPI C compiler is not defined")
  endif()

  if(NOT DEFINED MPI_CXX_COMPILER)
    message(FATAL_ERROR "MPI C++ compiler is not defined")
  endif()

  # working directory
  if(NOT DEFINED TIMPI_WORKING_DIR)
    set(TIMPI_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/timpi)
  endif()

  # install directory
  if(NOT DEFINED TIMPI_INSTALL_DIR)
    set(TIMPI_INSTALL_DIR ${FETCHCONTENT_BASE_DIR}/timpi)
  endif()

  message(STATUS "Configuring and installing TIMPI (${TIMPI_BUILD_TYPE})")
  set(TIMPI_WORKING_DIR ${NEML2_BINARY_DIR}/timpi)
  set(TIMPI_INSTALL_DIR ${FETCHCONTENT_BASE_DIR}/timpi)
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TIMPI_WORKING_DIR})
  execute_process(
    COMMAND ${CMAKE_COMMAND} \\
    -DCMAKE_C_COMPILER=${MPI_C_COMPILER} \\
    -DCMAKE_CXX_COMPILER=${MPI_CXX_COMPILER} \\
    -DVERSION=${TIMPI_VERSION} \\
    -DBASE_DIR=${TIMPI_INSTALL_DIR} \\
    -DBUILD_TYPE=${TIMPI_BUILD_TYPE} \\
    ${CMAKE_CURRENT_SOURCE_DIR}/timpi
    WORKING_DIRECTORY ${TIMPI_WORKING_DIR}
    OUTPUT_QUIET OUTPUT_FILE ${TIMPI_WORKING_DIR}/configure.log
    ERROR_QUIET ERROR_FILE ${TIMPI_WORKING_DIR}/configure.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${TIMPI_WORKING_DIR}
    OUTPUT_QUIET OUTPUT_FILE ${TIMPI_WORKING_DIR}/build.log
    ERROR_QUIET ERROR_FILE ${TIMPI_WORKING_DIR}/build.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  install(DIRECTORY
    ${TIMPI_INSTALL_DIR}/include/timpi
    TYPE INCLUDE
  )

  # Export variables
  set(TIMPI_FOUND TRUE)
  set(TIMPI_INCLUDE_DIRS ${TIMPI_INSTALL_DIR}/include)
  set(TIMPI_LIBRARIES timpi_${TIMPI_BUILD_TYPE})
  set(TIMPI_LINK_DIRS ${TIMPI_INSTALL_DIR}/lib)

  # Export interface
  add_library(TIMPI INTERFACE)
  target_include_directories(TIMPI SYSTEM INTERFACE ${TIMPI_INCLUDE_DIRS})
  target_link_directories(TIMPI INTERFACE ${TIMPI_LINK_DIRS})
  target_link_libraries(TIMPI INTERFACE ${TIMPI_LIBRARIES})
endif()
