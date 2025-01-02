include(NEML2UnityGroup)

file(GLOB_RECURSE srcs *.cxx)
add_executable(unit_tests ${srcs})
set_target_properties(unit_tests PROPERTIES INSTALL_RPATH "${EXEC_DIR}/../lib;${Torch_LINK_DIRECTORIES}")

target_compile_options(unit_tests PRIVATE -Wall -Wextra -pedantic)
register_unity_group(unit_tests .)
target_link_libraries(unit_tests PRIVATE testutils Catch2::Catch2WithMain)

install(TARGETS unit_tests COMPONENT Development)
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/unit
  COMPONENT Development
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
