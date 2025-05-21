# ------------------------------------------------------------------------------
# argparse
# ------------------------------------------------------------------------------
include(DownloadArgparse)
add_dependencies(deps argparse)

# ----------------------------------------------------------------------------
# CPU Profiler
# ----------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
  find_package(Gperftools)

  if(NOT Gperftools_FOUND)
    include(DownloadGperftools)
    set(Gperftools_ROOT ${GPERFTOOLS_INSTALL_DIR} CACHE PATH "gperftools root directory" FORCE)
    find_package(Gperftools REQUIRED)
  endif()

  cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${Gperftools_LINK_DIR} Gperftools_DOWNLOADED)

  if(Gperftools_DOWNLOADED)
    # libneml2 does not need gperftools headers
  endif()
endif()

# ------------------------------------------------------------------------------
# runner
# ------------------------------------------------------------------------------
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cxx)
add_executable(runner ${srcs})

target_compile_options(runner PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(runner PRIVATE neml2 argparse)

if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
  target_link_libraries(runner PRIVATE Gperftools::profiler)
endif()

set_target_properties(runner PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS runner COMPONENT libneml2-bin)
install(DIRECTORY benchmark
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/neml2
  COMPONENT libneml2-bin
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)

# ------------------------------------------------------------------------------
# tests
# ------------------------------------------------------------------------------
set(BENCHMARK_DIR ${NEML2_SOURCE_DIR}/runner/benchmark)
list(APPEND BENCHMARK_ARGS "driver")
list(APPEND BENCHMARK_ARGS "nbatch=1")
list(APPEND BENCHMARK_ARGS "device=cpu")

macro(add_benchmark name)
  add_test(NAME ${name} COMMAND runner ${BENCHMARK_DIR}/${name}/model.i ${BENCHMARK_ARGS})
endmacro()

add_benchmark(elasticity)
add_benchmark(radret)
add_benchmark(isoharden)
add_benchmark(chaboche2)
add_benchmark(chaboche4)
add_benchmark(chaboche6)
add_benchmark(gtntheig)
add_benchmark(scpcoup)
add_benchmark(scpdecoup)
add_benchmark(scpdecoupexp)
add_benchmark(tcpsingle)
add_benchmark(tcprandom)
