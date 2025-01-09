file(GLOB_RECURSE srcs *.cxx)
add_executable(unit_tests ${srcs})
target_link_libraries(unit_tests PRIVATE testutils Catch2::Catch2)
target_compile_options(unit_tests PRIVATE -Wall -Wextra -pedantic)
set_target_properties(unit_tests PROPERTIES LINK_WHAT_YOU_USE TRUE INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS unit_tests COMPONENT libneml2)
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/neml2/unit
  COMPONENT libneml2
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
