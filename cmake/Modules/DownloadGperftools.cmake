# Download Gperftools source code from GitHub, then configure, build and install Gperftools.
#
# Influential variables:
# - GPERFTOOLS_VERSION
# - CMAKE_BINARY_DIR
# - GLIBCXX_USE_CXX11_ABI
#
# Output variables are those defined by FetchContent such as:
# - gperftools_POPULATED
# - gperftools_SOURCE_DIR
#
# as well as the installation directory (where Gperftools is installed):
# - GPERFTOOLS_INSTALL_DIR

set(Gperftools_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/gperftools-src)
set(Gperftools_SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/gperftools-subbuild)
set(Gperftools_WORKING_DIR ${CMAKE_BINARY_DIR}/_deps/gperftools-src)
set(GPERFTOOLS_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/gperftools CACHE PATH "Gperftools installation directory" FORCE)
include(FetchContent)

message(STATUS "Downloading/updating Gperftools")
FetchContent_Populate(
  gperftools
  QUIET
  GIT_REPOSITORY https://github.com/gperftools/gperftools.git
  GIT_TAG gperftools-${GPERFTOOLS_VERSION}
  SOURCE_DIR ${Gperftools_SOURCE_DIR}
  BINARY_DIR ${Gperftools_WORKING_DIR}
  SUBBUILD_DIR ${Gperftools_SUBBUILD_DIR}
)

# cxx11 abi
if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  message(WARNING "GLIBCXX_USE_CXX11_ABI not defined, assuming 1")
  set(GLIBCXX_USE_CXX11_ABI 1)
endif()

# install
include(ProcessorCount)
ProcessorCount(NPROCS)
configure_file(${CMAKE_CURRENT_LIST_DIR}/InstallGperftools.sh.in ${Gperftools_WORKING_DIR}/InstallGperftools.sh)
message(STATUS "Installing Gperftools")
execute_process(
  COMMAND bash -c ./InstallGperftools.sh
  WORKING_DIRECTORY ${Gperftools_WORKING_DIR}
  OUTPUT_QUIET OUTPUT_FILE configure.log
  ERROR_QUIET ERROR_FILE configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
