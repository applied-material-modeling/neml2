# Download TIMPI source code from GitHub, then configure, build and install TIMPI.
#
# Influential variables:
# - TIMPI_VERSION
# - CMAKE_BUILD_TYPE
# - CMAKE_BINARY_DIR
# - GLIBCXX_USE_CXX11_ABI
#
# Output variables are those defined by FetchContent such as:
# - timpi_POPULATED
# - timpi_SOURCE_DIR
#
# as well as the installation directory (where TIMPI is installed):
# - TIMPI_INSTALL_DIR

include(FetchContent)
FetchContent_Declare(
  timpi
  GIT_REPOSITORY https://github.com/libMesh/TIMPI.git
  GIT_TAG v${TIMPI_VERSION}
)

message(STATUS "Downloading/updating TIMPI")
FetchContent_MakeAvailable(timpi)

# build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(TIMPI_BUILD_TYPE "dbg")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(TIMPI_BUILD_TYPE "devel")
else()
  set(TIMPI_BUILD_TYPE "opt")
endif()

# cxx11 abi
if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  message(WARNING "GLIBCXX_USE_CXX11_ABI not defined, assuming 1")
  set(GLIBCXX_USE_CXX11_ABI 1)
endif()

# install
set(TIMPI_WORKING_DIR ${CMAKE_BINARY_DIR}/_deps/timpi-build)
set(TIMPI_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/timpi CACHE PATH "TIMPI installation directory")
include(ProcessorCount)
ProcessorCount(NPROCS)
configure_file(${CMAKE_CURRENT_LIST_DIR}/InstallTIMPI.sh.in ${TIMPI_WORKING_DIR}/InstallTIMPI.sh)
message(STATUS "Installing TIMPI")
execute_process(
  COMMAND bash -c ./InstallTIMPI.sh
  WORKING_DIRECTORY ${TIMPI_WORKING_DIR}
  OUTPUT_QUIET OUTPUT_FILE configure.log
  ERROR_QUIET ERROR_FILE configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
