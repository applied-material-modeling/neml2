project(wasp)
cmake_minimum_required(VERSION 3.28)
include(ExternalProject)

ExternalProject_Add(
  timpi
  GIT_REPOSITORY https://github.com/libMesh/TIMPI.git
  GIT_TAG v${VERSION}
  PREFIX ${PREFIX}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env METHODS=${BUILD_TYPE} ${PREFIX}/src/timpi/configure --prefix=${INSTALL_PREFIX} --enable-shared=no --enable-static=yes
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  TEST_EXCLUDE_FROM_MAIN ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)
