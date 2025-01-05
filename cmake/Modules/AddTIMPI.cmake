include(FetchContent)
include(ExternalProject)

ExternalProject_Add(
  timpi_ext
  GIT_REPOSITORY https://github.com/libMesh/TIMPI.git
  GIT_TAG v${TIMPI_VERSION}
  PREFIX ${FETCHCONTENT_BASE_DIR}/timpi
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env METHODS=${timpi_BUILD_TYPE} ${FETCHCONTENT_BASE_DIR}/timpi/src/timpi_ext/configure --prefix=${NEML2_BINARY_DIR}/timpi/install --enable-shared=no --enable-static=yes
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  TEST_EXCLUDE_FROM_MAIN ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)
