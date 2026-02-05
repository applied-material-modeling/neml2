# ------------------------------------------------------------------------------
# runner
# ------------------------------------------------------------------------------
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cxx)
add_executable(runner ${srcs})
set_target_properties(runner PROPERTIES
  DISABLE_PRECOMPILE_HEADERS ON
  OUTPUT_NAME neml2
)

target_compile_options(runner PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(runner PRIVATE neml2 argparse::argparse)

if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
  target_link_libraries(runner PRIVATE gperftools::profiler)
endif()

if(NEML2_WHEEL)
  set_target_properties(runner PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/../lib;${INSTALL_REL_PATH}/../../torch/lib")
else()
  set_target_properties(runner PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/../lib;${torch_LINK_DIR}")
endif()

install(TARGETS runner COMPONENT libneml2-bin)
install(DIRECTORY benchmark
  DESTINATION share/neml2
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
if(NEML2_TESTS)
  set(BENCHMARK_DIR ${NEML2_SOURCE_DIR}/runner/benchmark)
  list(APPEND BENCHMARK_ARGS "driver")
  list(APPEND BENCHMARK_ARGS "nbatch=1")
  list(APPEND BENCHMARK_ARGS "device=cpu")

  macro(add_benchmark name)
    add_test(NAME ${name} COMMAND neml2 run ${BENCHMARK_DIR}/${name}/model.i ${BENCHMARK_ARGS})
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
endif()
