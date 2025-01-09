file(GLOB_RECURSE srcs *.cxx)
add_executable(regression_tests ${srcs})
set_target_properties(regression_tests PROPERTIES INSTALL_RPATH "${EXEC_DIR}/../lib;${Torch_LINK_DIRECTORIES}")

target_compile_options(regression_tests PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(regression_tests PRIVATE testutils Catch2::Catch2WithMain)
set_target_properties(regression_tests PROPERTIES LINK_WHAT_YOU_USE TRUE)

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
