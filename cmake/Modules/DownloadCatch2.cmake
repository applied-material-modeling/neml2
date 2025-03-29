# Download Catch2 source code from GitHub, then configure, build, and install HIT.
#
# Influential variables:
# - CATCH2_VERSION
#
# Output variables are those defined by FetchContent such as:
# - catch2_POPULATED
# - catch2_SOURCE_DIR

set(CATCH2_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/catch2-src)
set(CATCH2_SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/catch2-subbuild)
set(CATCH2_WORKING_DIR ${CMAKE_BINARY_DIR}/_deps/catch2-build)
set(CATCH2_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/catch2 CACHE PATH "Catch2 installation directory" FORCE)

include(FetchContent)
message(STATUS "Downloading/updating Catch2")
FetchContent_Populate(
  catch2
  QUIET
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v${CATCH2_VERSION}
  SOURCE_DIR ${CATCH2_SOURCE_DIR}
  BINARY_DIR ${CATCH2_WORKING_DIR}
  SUBBUILD_DIR ${CATCH2_SUBBUILD_DIR}
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

  set(CATCH2_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
endif()

# install
message(STATUS "Installing Catch2")
execute_process(
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${CATCH2_CXX_FLAGS} -G${CMAKE_GENERATOR} -B${catch2_BINARY_DIR} -S.
  WORKING_DIRECTORY ${catch2_SOURCE_DIR}
  OUTPUT_QUIET OUTPUT_FILE configure.log
  ERROR_QUIET ERROR_FILE configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${catch2_BINARY_DIR}
  WORKING_DIRECTORY ${catch2_SOURCE_DIR}
  OUTPUT_QUIET OUTPUT_FILE build.log
  ERROR_QUIET ERROR_FILE build.err
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND ${CMAKE_COMMAND} --install ${catch2_BINARY_DIR} --prefix=${CATCH2_INSTALL_DIR}
  WORKING_DIRECTORY ${catch2_SOURCE_DIR}
  OUTPUT_QUIET OUTPUT_FILE install.log
  ERROR_QUIET ERROR_FILE install.err
  COMMAND_ERROR_IS_FATAL ANY
)
