cmake_minimum_required(VERSION 3.28)
project(hit_ext LANGUAGES C CXX)
include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${BASE_DIR})
FetchContent_Declare(
  hit
  GIT_REPOSITORY https://github.com/idaholab/hit.git
  GIT_TAG ${VERSION}
)
FetchContent_MakeAvailable(hit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE ${BUILD_TYPE})

add_library(hit STATIC
  ${hit_SOURCE_DIR}/src/hit/parse.cc
  ${hit_SOURCE_DIR}/src/hit/lex.cc
  ${hit_SOURCE_DIR}/src/hit/braceexpr.cc
)

message(STATUS "WASP_INCLUDE_DIRS: ${WASP_INCLUDE_DIRS}")
message(STATUS "WASP_LIBRARIES: ${WASP_LIBRARIES}")
message(STATUS "WASP_LINK_DIRS: ${WASP_LINK_DIRS}")

set_target_properties(hit PROPERTIES UNITY_BUILD OFF POSITION_INDEPENDENT_CODE ON EXPORT_COMPILE_COMMANDS ON)
target_compile_definitions(hit PRIVATE _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})
target_include_directories(hit SYSTEM PUBLIC ${hit_SOURCE_DIR}/include ${WASP_INCLUDE_DIRS})
target_link_libraries(hit PRIVATE ${WASP_LIBRARIES})
target_link_directories(hit PRIVATE ${WASP_LINK_DIRS})

target_sources(hit
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${hit_SOURCE_DIR}/include
  FILES
  ${hit_SOURCE_DIR}/include/hit/braceexpr.h
  ${hit_SOURCE_DIR}/include/hit/hit.h
  ${hit_SOURCE_DIR}/include/hit/lex.h
  ${hit_SOURCE_DIR}/include/hit/parse.h
)
install(TARGETS hit COMPONENT Development FILE_SET HEADERS)
