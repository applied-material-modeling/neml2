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

# Work dispatcher
option(NEML2_WORK_DISPATCHER "Enable NEML2 work dispatcher" OFF)

if(NEML2_WORK_DISPATCHER)
  find_package(MPI REQUIRED)

  if(MPI_CXX_FOUND)
    set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
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

  message(STATUS "Configuring TIMPI")
  set(timpi_BUILD_DIR ${NEML2_BINARY_DIR}/timpi/build)
  set(timpi_INSTALL_DIR ${NEML2_BINARY_DIR}/timpi/install)
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${timpi_BUILD_DIR})
  execute_process(
    COMMAND ${CMAKE_COMMAND} \\
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \\
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \\
    -DVERSION=${TIMPI_VERSION} \\
    -DPREFIX=${FETCHCONTENT_BASE_DIR}/timpi \\
    -DBUILD_TYPE=${timpi_BUILD_TYPE} \\
    -DINSTALL_PREFIX=${timpi_INSTALL_DIR} \\
    ${NEML2_SOURCE_DIR}/cmake/timpi
    WORKING_DIRECTORY ${timpi_BUILD_DIR}
    OUTPUT_QUIET
    OUTPUT_FILE ${timpi_BUILD_DIR}/configure.log
    ERROR_QUIET
    ERROR_FILE ${timpi_BUILD_DIR}/configure.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${timpi_BUILD_DIR}
    WORKING_DIRECTORY ${timpi_BUILD_DIR}
    OUTPUT_QUIET
    OUTPUT_FILE ${timpi_BUILD_DIR}/build.log
    ERROR_QUIET
    ERROR_FILE ${timpi_BUILD_DIR}/build.err
    COMMAND_ERROR_IS_FATAL ANY
  )

  install(DIRECTORY
    ${timpi_INSTALL_DIR}/include/timpi
    TYPE INCLUDE
  )
  target_include_directories(neml2 SYSTEM PUBLIC ${timpi_INSTALL_DIR}/include)
  target_link_libraries(neml2 PRIVATE timpi_${timpi_BUILD_TYPE})
  target_link_directories(neml2 PRIVATE ${timpi_INSTALL_DIR}/lib)
endif()

# wasp
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(wasp_BUILD_TYPE "Debug")
else()
  set(wasp_BUILD_TYPE "RelWithDebInfo")
endif()

message(STATUS "Configuring WASP")
set(wasp_BUILD_DIR ${NEML2_BINARY_DIR}/wasp/build)
set(wasp_INSTALL_DIR ${NEML2_BINARY_DIR}/wasp/install)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${wasp_BUILD_DIR})
execute_process(
  COMMAND ${CMAKE_COMMAND} \\
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \\
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \\
  -DVERSION=${WASP_VERSION} \\
  -DPREFIX=${FETCHCONTENT_BASE_DIR}/wasp \\
  -DBUILD_TYPE=${wasp_BUILD_TYPE} \\
  -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \\
  -DINSTALL_PREFIX=${wasp_INSTALL_DIR} \\
  ${NEML2_SOURCE_DIR}/cmake/wasp
  WORKING_DIRECTORY ${wasp_BUILD_DIR}
  OUTPUT_QUIET
  OUTPUT_FILE ${wasp_BUILD_DIR}/configure.log
  ERROR_QUIET
  ERROR_FILE ${wasp_BUILD_DIR}/configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${wasp_BUILD_DIR}
  OUTPUT_QUIET
  OUTPUT_FILE ${wasp_BUILD_DIR}/build.log
  ERROR_QUIET
  ERROR_FILE ${wasp_BUILD_DIR}/build.err
  COMMAND_ERROR_IS_FATAL ANY
)

# HIT
message(STATUS "Configuring HIT")
FetchContent_MakeAvailable(hit)

add_library(hit SHARED
  ${hit_SOURCE_DIR}/src/hit/parse.cc
  ${hit_SOURCE_DIR}/src/hit/lex.cc
  ${hit_SOURCE_DIR}/src/hit/braceexpr.cc
)
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
