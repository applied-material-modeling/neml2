# Download HWLOC source code from GitHub, then configure, build and install HWLOC.
#
# Influential variables:
# - HWLOC_VERSION
# - CMAKE_BINARY_DIR
# - GLIBCXX_USE_CXX11_ABI
#
# Output variables are those defined by FetchContent such as:
# - hwloc_POPULATED
# - hwloc_SOURCE_DIR
#
# as well as the installation directory (where HWLOC is installed):
# - HWLOC_INSTALL_DIR

include(FetchContent)
FetchContent_Declare(
  hwloc
  GIT_REPOSITORY https://github.com/open-mpi/hwloc.git
  GIT_TAG v${HWLOC_VERSION}
)

message(STATUS "Downloading/updating HWLOC")
FetchContent_MakeAvailable(hwloc)

# cxx11 abi
if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  message(WARNING "GLIBCXX_USE_CXX11_ABI not defined, assuming 1")
  set(GLIBCXX_USE_CXX11_ABI 1)
endif()

# install
set(HWLOC_WORKING_DIR ${CMAKE_BINARY_DIR}/_deps/hwloc-build)
set(HWLOC_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/hwloc CACHE PATH "HWLOC installation directory")
include(ProcessorCount)
ProcessorCount(NPROCS)
configure_file(${CMAKE_CURRENT_LIST_DIR}/InstallHwloc.sh.in ${HWLOC_WORKING_DIR}/InstallHwloc.sh)
message(STATUS "Installing HWLOC")
execute_process(
  COMMAND bash -c ./InstallHwloc.sh
  WORKING_DIRECTORY ${HWLOC_WORKING_DIR}
  OUTPUT_QUIET OUTPUT_FILE configure.log
  ERROR_QUIET ERROR_FILE configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
