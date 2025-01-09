file(GLOB_RECURSE srcs *.cxx)
add_executable(dispatcher_tests ${srcs})
set_target_properties(dispatcher_tests PROPERTIES INSTALL_RPATH "${EXEC_DIR}/../lib;${Torch_LINK_DIRECTORIES}")

target_compile_options(dispatcher_tests PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(dispatcher_tests PRIVATE neml2_dispatcher Catch2::Catch2)

install(TARGETS dispatcher_tests COMPONENT Development)
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dispatcher
  COMPONENT Development
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
