file(GLOB_RECURSE srcs *.cxx)
add_executable(verification_tests ${srcs})
target_link_libraries(verification_tests PRIVATE testutils Catch2::Catch2)
target_compile_options(verification_tests PRIVATE -Wall -Wextra -pedantic)
set_target_properties(verification_tests PROPERTIES LINK_WHAT_YOU_USE TRUE INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS verification_tests COMPONENT libneml2)
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/neml2/verification
  COMPONENT libneml2
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
