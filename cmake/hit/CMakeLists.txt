cmake_minimum_required(VERSION 3.28)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
project(hit_ext LANGUAGES CXX)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FETCHCONTENT_BASE_DIR ${PREFIX})

FetchContent_Declare(
  hit
  GIT_REPOSITORY https://github.com/idaholab/hit.git
  GIT_TAG ${VERSION}
)
FetchContent_MakeAvailable(hit)

set(CMAKE_BUILD_TYPE ${BUILD_TYPE})

add_library(hit STATIC
  ${hit_SOURCE_DIR}/src/hit/parse.cc
  ${hit_SOURCE_DIR}/src/hit/lex.cc
  ${hit_SOURCE_DIR}/src/hit/braceexpr.cc
)

set_target_properties(hit PROPERTIES UNITY_BUILD OFF)
target_compile_options(hit PRIVATE -fPIC) # Setting POSITION_INDEPENDENT_CODE has no effect -- I believe this is a CMake bug.
target_compile_definitions(hit PRIVATE _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})
target_include_directories(hit SYSTEM PUBLIC ${hit_SOURCE_DIR}/include ${wasp_INSTALL_DIR}/include)
target_link_libraries(hit PRIVATE waspcore wasphit)
target_link_directories(hit PRIVATE ${wasp_INSTALL_DIR}/lib)

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
