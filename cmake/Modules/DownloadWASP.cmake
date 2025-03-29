# Download WASP source code from GitHub, then configure, build and install WASP.
#
# Influential variables:
# - WASP_VERSION
# - CMAKE_BINARY_DIR
# - COMPILE_DEFINITIONS
#
# Output variables are those defined by FetchContent such as:
# - wasp_POPULATED
# - wasp_SOURCE_DIR
#
# as well as the installation directory (where WASP is installed):
# - WASP_INSTALL_DIR

set(WASP_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/wasp-src)
set(WASP_SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/wasp-subbuild)
set(WASP_WORKING_DIR ${CMAKE_BINARY_DIR}/_deps/wasp-build)
set(WASP_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/wasp CACHE PATH "WASP installation directory" FORCE)
include(FetchContent)

message(STATUS "Downloading/updating WASP")
FetchContent_Populate(
  wasp
  QUIET
  GIT_REPOSITORY https://code.ornl.gov/neams-workbench/wasp.git
  GIT_TAG ${WASP_VERSION}
  SOURCE_DIR ${WASP_SOURCE_DIR}
  BINARY_DIR ${WASP_WORKING_DIR}
  SUBBUILD_DIR ${WASP_SUBBUILD_DIR}
)

# Check if the system is using glibc
if(NOT DEFINED HAS_GLIBC)
  include(CheckSymbolExists)
  check_symbol_exists(__GLIBC__ "features.h" HAS_GLIBC)
  set(HAS_GLIBC ${HAS_GLIBC} CACHE INTERNAL "System has glibc" FORCE)
endif()

# cxx11 abi
if(HAS_GLIBC)
  if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
    message(WARNING "GLIBCXX_USE_CXX11_ABI not defined, assuming 1")
    set(GLIBCXX_USE_CXX11_ABI 1)
  endif()

  set(WASP_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
endif()

# install
configure_file(${CMAKE_CURRENT_LIST_DIR}/InstallWASP.sh.in ${WASP_WORKING_DIR}/InstallWASP.sh)
message(STATUS "Installing WASP")
execute_process(
  COMMAND bash -c ./InstallWASP.sh
  WORKING_DIRECTORY ${WASP_WORKING_DIR}
  OUTPUT_QUIET OUTPUT_FILE configure.log
  ERROR_QUIET ERROR_FILE configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
