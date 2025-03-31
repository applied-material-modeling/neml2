# Download HIT source code from GitHub, then configure, build, and install HIT.
#
# Influential variables:
# - HIT_VERSION
# - HIT_SOURCE_DIR, the path to the HIT source directory (if already downloaded)
#
# Output variables are those defined by FetchContent such as:
# - hit_POPULATED
# - hit_SOURCE_DIR

find_package(WASP REQUIRED)

if(NOT HIT_SOURCE_DIR)
  include(FetchContent)
  FetchContent_Declare(
    hit
    GIT_REPOSITORY https://github.com/idaholab/hit.git
    GIT_TAG ${HIT_VERSION}
  )

  message(STATUS "Downloading/updating HIT")
  FetchContent_MakeAvailable(hit)

  set(HIT_SOURCE_DIR ${hit_SOURCE_DIR} CACHE PATH "HIT source directory" FORCE)
  set(HIT_BINARY_DIR ${hit_BINARY_DIR} CACHE PATH "HIT binary directory" FORCE)
else()
  set(HIT_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/hit-build CACHE PATH "HIT binary directory" FORCE)
endif()

# install
set(HIT_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/hit CACHE PATH "HIT installation directory" FORCE)
file(COPY ${HIT_SOURCE_DIR}/include DESTINATION ${HIT_BINARY_DIR} FILES_MATCHING PATTERN "*.h")
file(COPY ${HIT_SOURCE_DIR}/src DESTINATION ${HIT_BINARY_DIR} FILES_MATCHING PATTERN "*.cpp" PATTERN "*.cc")
configure_file(${CMAKE_CURRENT_LIST_DIR}/InstallHIT.cmake.in ${HIT_BINARY_DIR}/CMakeLists.txt)
message(STATUS "Installing HIT")
execute_process(
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=${WASP_CXX_FLAGS} -G${CMAKE_GENERATOR} -B${HIT_BINARY_DIR} -S.
  WORKING_DIRECTORY ${HIT_BINARY_DIR}
  OUTPUT_QUIET OUTPUT_FILE configure.log
  ERROR_QUIET ERROR_FILE configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${HIT_BINARY_DIR}
  WORKING_DIRECTORY ${HIT_BINARY_DIR}
  OUTPUT_QUIET OUTPUT_FILE build.log
  ERROR_QUIET ERROR_FILE build.err
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND ${CMAKE_COMMAND} --install ${HIT_BINARY_DIR} --prefix=${HIT_INSTALL_DIR}
  WORKING_DIRECTORY ${HIT_BINARY_DIR}
  OUTPUT_QUIET OUTPUT_FILE install.log
  ERROR_QUIET ERROR_FILE install.err
  COMMAND_ERROR_IS_FATAL ANY
)
