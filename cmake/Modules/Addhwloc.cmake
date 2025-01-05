include(FetchContent)
include(ExternalProject)

ExternalProject_Add(
  hwloc_ext
  GIT_REPOSITORY https://github.com/open-mpi/hwloc.git
  GIT_TAG hwloc-${HWLOC_VERSION}
  PREFIX ${FETCHCONTENT_BASE_DIR}/hwloc
  PATCH_COMMAND ${FETCHCONTENT_BASE_DIR}/hwloc/src/hwloc_ext/autogen.sh
  CONFIGURE_COMMAND ${FETCHCONTENT_BASE_DIR}/hwloc/src/hwloc_ext/configure --prefix=${NEML2_BINARY_DIR}/hwloc/install
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  TEST_EXCLUDE_FROM_MAIN ON
  LOG_DOWNLOAD ON
  LOG_PATCH ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)
