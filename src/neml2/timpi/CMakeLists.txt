cmake_minimum_required(VERSION 3.28)
project(timpi_ext)
include(ExternalProject)

ExternalProject_Add(
  timpi
  GIT_REPOSITORY https://github.com/libMesh/TIMPI.git
  GIT_TAG v${VERSION}
  PREFIX ${BASE_DIR}
  UPDATE_DISCONNECTED TRUE
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} METHODS=${BUILD_TYPE} ${BASE_DIR}/src/timpi/configure --prefix=${BASE_DIR} --enable-shared=no --enable-static=yes
  TEST_EXCLUDE_FROM_MAIN ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)
