include(NEML2UnityGroup)

message(STATUS "Configuring argparse")
FetchContent_MakeAvailable(argparse)

file(GLOB_RECURSE srcs src/*.cxx)
add_executable(runner ${srcs})
set_target_properties(runner PROPERTIES INSTALL_RPATH "${EXEC_DIR}/../lib;${Torch_LINK_DIRECTORIES}")
register_unity_group(runner .)
target_compile_options(runner PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(runner PRIVATE neml2 argparse)

install(TARGETS runner)
install(DIRECTORY benchmark
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/benchmark
  FILES_MATCHING
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "*.vtest"
  PATTERN "*.xml"
)
