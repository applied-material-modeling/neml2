include(NEML2UnityGroup)

file(GLOB_RECURSE srcs *.cxx)
add_executable(verification_tests ${srcs})
set_target_properties(verification_tests PROPERTIES INSTALL_RPATH "${EXEC_DIR}/../lib;${Torch_LINK_DIRECTORIES}")

target_compile_options(verification_tests PRIVATE -Wall -Wextra -pedantic)
register_unity_group(verification_tests .)
target_link_libraries(verification_tests PRIVATE testutils Catch2::Catch2WithMain)

install(TARGETS verification_tests COMPONENT Development)
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/verification
  COMPONENT Development
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
