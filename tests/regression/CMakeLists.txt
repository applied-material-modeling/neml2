include(NEML2UnityGroup)

file(GLOB_RECURSE srcs *.cxx)
add_executable(regression_tests ${srcs})
set_target_properties(regression_tests PROPERTIES INSTALL_RPATH "${EXEC_DIR}/../lib;${Torch_LINK_DIRECTORIES}")

target_compile_options(regression_tests PRIVATE -Wall -Wextra -pedantic -Werror)
register_unity_group(regression_tests .)
target_link_libraries(regression_tests PRIVATE testutils Catch2::Catch2WithMain)

option(NEML2_REGRESSION_AS_PROFILER "Additionally link the regression test executable against gperftools profiler" OFF)

if(NEML2_REGRESSION_AS_PROFILER)
  FetchContent_MakeAvailable(gperftools)
  set_target_properties(regression_tests PROPERTIES LINK_WHAT_YOU_USE TRUE)
  target_link_libraries(regression_tests PRIVATE profiler)
endif()

install(TARGETS regression_tests COMPONENT Development)
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/regression
  COMPONENT Development
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
