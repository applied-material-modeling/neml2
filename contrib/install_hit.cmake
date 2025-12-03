cmake_minimum_required(VERSION 3.26)
project(hit LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_library(hit SHARED
  src/hit/parse.cc
  src/hit/lex.cc
  src/hit/braceexpr.cc
)
target_sources(hit
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS include
  FILES
  include/hit/braceexpr.h
  include/hit/hit.h
  include/hit/lex.h
  include/hit/parse.h
)
target_include_directories(hit PUBLIC @wasp_hit_INCLUDE_DIR@)
target_link_directories(hit PUBLIC @wasp_hit_LINK_DIR@)
target_link_libraries(hit PUBLIC waspcore wasphit)
set_target_properties(hit PROPERTIES INSTALL_RPATH "@INSTALL_REL_PATH@")

install(TARGETS hit LIBRARY)
install(TARGETS hit FILE_SET HEADERS)
