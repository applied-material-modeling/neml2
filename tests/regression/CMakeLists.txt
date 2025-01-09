file(GLOB_RECURSE srcs *.cxx)
add_executable(regression_tests ${srcs})
target_link_libraries(regression_tests PRIVATE testutils Catch2::Catch2)
target_compile_options(regression_tests PRIVATE -Wall -Wextra -pedantic)
set_target_properties(regression_tests PROPERTIES LINK_WHAT_YOU_USE TRUE INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS regression_tests COMPONENT libneml2)
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/neml2/regression
  COMPONENT libneml2
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
