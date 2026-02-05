# ------------------------------------------------------------------------------
# catch-all target for all tools
# ------------------------------------------------------------------------------
add_custom_target(tools)

# ------------------------------------------------------------------------------
# helper function to add tools
# ------------------------------------------------------------------------------
function(add_tool name)
  add_executable(${name} ${name}.cxx)
  target_compile_options(${name} PRIVATE -Wall -Wextra -pedantic)
  target_link_libraries(${name} PRIVATE neml2 argparse::argparse)
  if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
    target_link_libraries(${name} PRIVATE gperftools::profiler)
  endif()

  target_sources(${name}
    PRIVATE
    FILE_SET HEADERS
    BASE_DIRS
    ${NEML2_SOURCE_DIR}/include/tools
    FILES
    ${NEML2_SOURCE_DIR}/include/tools/utils.h
  )

  set_target_properties(${name} PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)
  if(NEML2_WHEEL AND NOT SKBUILD_STATE STREQUAL "editable")
    set_target_properties(${name} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/../lib;${INSTALL_REL_PATH}/../../torch/lib")
  else()
    set_target_properties(${name} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/../lib;${torch_LINK_DIR}")
  endif()

  add_dependencies(tools ${name})
  install(TARGETS ${name} DESTINATION ${INSTALL_BINDIR} COMPONENT libneml2-bin)
endfunction()

# ------------------------------------------------------------------------------
# tools
# ------------------------------------------------------------------------------
add_tool(diagnose)
add_tool(run)
add_tool(inspect)
add_tool(time)
add_tool(syntax)

# ------------------------------------------------------------------------------
# benchmarks
# ------------------------------------------------------------------------------
install(DIRECTORY benchmark
  DESTINATION share/neml2
  COMPONENT libneml2-bin
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)

if(NEML2_TESTS)
  set(BENCHMARK_DIR ${NEML2_SOURCE_DIR}/benchmark)
  list(APPEND BENCHMARK_ARGS "driver")
  list(APPEND BENCHMARK_ARGS "nbatch=1")
  list(APPEND BENCHMARK_ARGS "device=cpu")

  macro(add_benchmark name)
    add_test(NAME ${name} COMMAND run ${BENCHMARK_DIR}/${name}/model.i ${BENCHMARK_ARGS})
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
