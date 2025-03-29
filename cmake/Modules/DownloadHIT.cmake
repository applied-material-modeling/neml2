# Download HIT source code from GitHub, then configure, build, and install HIT.
#
# Influential variables:
# - HIT_VERSION
#
# Output variables are those defined by FetchContent such as:
# - hit_POPULATED
# - hit_SOURCE_DIR

find_package(WASP REQUIRED)

include(FetchContent)
FetchContent_Declare(
  hit
  GIT_REPOSITORY https://github.com/idaholab/hit.git
  GIT_TAG ${HIT_VERSION}
)

message(STATUS "Downloading/updating HIT")
FetchContent_MakeAvailable(hit)

# install
set(HIT_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/hit CACHE PATH "HIT installation directory" FORCE)
configure_file(${CMAKE_CURRENT_LIST_DIR}/InstallHIT.cmake.in ${hit_SOURCE_DIR}/CMakeLists.txt)
message(STATUS "Installing HIT")
execute_process(
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=${WASP_CXX_FLAGS} -G${CMAKE_GENERATOR} -B${hit_BINARY_DIR} -S.
  WORKING_DIRECTORY ${hit_SOURCE_DIR}
  OUTPUT_QUIET OUTPUT_FILE configure.log
  ERROR_QUIET ERROR_FILE configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${hit_BINARY_DIR}
  WORKING_DIRECTORY ${hit_SOURCE_DIR}
  OUTPUT_QUIET OUTPUT_FILE build.log
  ERROR_QUIET ERROR_FILE build.err
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND ${CMAKE_COMMAND} --install ${hit_BINARY_DIR} --prefix=${HIT_INSTALL_DIR}
  WORKING_DIRECTORY ${hit_SOURCE_DIR}
  OUTPUT_QUIET OUTPUT_FILE install.log
  ERROR_QUIET ERROR_FILE install.err
  COMMAND_ERROR_IS_FATAL ANY
)
