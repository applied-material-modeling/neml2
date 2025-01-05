include(NEML2UnityGroup)

# Add all the source files
file(GLOB_RECURSE SRCS *.cxx)
add_library(neml2 SHARED ${SRCS})
set_target_properties(neml2
  PROPERTIES
  INSTALL_RPATH "${EXEC_DIR};${Torch_LINK_DIRECTORIES}"
  LINK_WHAT_YOU_USE TRUE
)

# NEML2 (private) compile options
target_compile_options(neml2 PRIVATE -Wall -Wextra -pedantic -Werror)

# Group source files together if UNITY build is requested
register_unity_group(neml2 .)

# NEML2 headers
file(GLOB_RECURSE _NEML2_HEADERS ${NEML2_SOURCE_DIR}/include/*.h)
target_include_directories(neml2 PUBLIC ${NEML2_SOURCE_DIR}/include ${NEML2_BINARY_DIR}/include)
target_sources(neml2
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS ${NEML2_SOURCE_DIR}/include
  FILES
  ${_NEML2_HEADERS}
)
install(TARGETS neml2 COMPONENT Development FILE_SET HEADERS)

# torch
target_include_directories(neml2 SYSTEM PUBLIC ${Torch_INCLUDE_DIRECTORIES})
target_link_directories(neml2 PUBLIC ${Torch_LINK_DIRECTORIES})
target_link_libraries(neml2 PUBLIC ${Torch_LIBRARIES})

# gperftools
option(NEML2_CPU_PROFILER "Link against gperftools libprofiler to enable CPU profiling" OFF)

if(NEML2_CPU_PROFILER)
  FetchContent_MakeAvailable(gperftools)
  target_link_libraries(neml2 PUBLIC profiler)
endif()

# HIT (WASP) for parsing
message(STATUS "Configuring HIT")
FetchContent_MakeAvailable(hit)
include(Addwasp)

add_library(hit SHARED
  ${hit_SOURCE_DIR}/src/hit/parse.cc
  ${hit_SOURCE_DIR}/src/hit/lex.cc
  ${hit_SOURCE_DIR}/src/hit/braceexpr.cc
)
add_dependencies(hit wasp_ext)
set(wasp_INSTALL_DIR ${NEML2_BINARY_DIR}/wasp/install)
install(DIRECTORY
  ${wasp_INSTALL_DIR}/include/waspcore
  ${wasp_INSTALL_DIR}/include/wasphit
  TYPE INCLUDE
)

set_target_properties(hit PROPERTIES UNITY_BUILD OFF)
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
target_link_libraries(neml2 PUBLIC hit)

# Work dispatcher
option(NEML2_WORK_DISPATCHER "Enable NEML2 work dispatcher" OFF)

if(NEML2_WORK_DISPATCHER)
  find_package(MPI REQUIRED)

  if(MPI_CXX_FOUND)
    target_link_libraries(neml2 PUBLIC MPI::MPI_CXX)
  else()
    message(FATAL_ERROR "MPI C++ compiler (required for work dispatcher) not found")
  endif()

  target_compile_definitions(neml2 PUBLIC NEML2_WORK_DISPATCHER)

  # TIMPI
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(timpi_BUILD_TYPE "dbg")
  else()
    set(timpi_BUILD_TYPE "opt")
  endif()

  include(AddTIMPI)
  add_dependencies(neml2 timpi_ext)
  set(timpi_INSTALL_DIR ${NEML2_BINARY_DIR}/timpi/install)
  install(DIRECTORY
    ${timpi_INSTALL_DIR}/include/timpi
    TYPE INCLUDE
  )
  target_include_directories(neml2 SYSTEM PUBLIC ${timpi_INSTALL_DIR}/include)
  target_link_libraries(neml2 PRIVATE timpi_${timpi_BUILD_TYPE})
  target_link_directories(neml2 PRIVATE ${timpi_INSTALL_DIR}/lib)

  # hwloc
  include(Addhwloc)
  add_dependencies(neml2 hwloc_ext)
  set(hwloc_INSTALL_DIR ${NEML2_BINARY_DIR}/hwloc/install)
  install(DIRECTORY
    ${hwloc_INSTALL_DIR}/include/hwloc
    TYPE INCLUDE
  )
  install(FILES
    ${hwloc_INSTALL_DIR}/include/hwloc.h
    TYPE INCLUDE
  )
  target_include_directories(neml2 SYSTEM PUBLIC ${hwloc_INSTALL_DIR}/include)
  target_link_libraries(neml2 PRIVATE hwloc)
  target_link_directories(neml2 PRIVATE ${hwloc_INSTALL_DIR}/lib)
endif()
