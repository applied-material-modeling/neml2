file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cxx)
add_executable(unit_tests ${srcs})
target_link_libraries(unit_tests PRIVATE testutils Catch2::Catch2)
target_compile_options(unit_tests PRIVATE -Wall -Wextra -pedantic)
set_target_properties(unit_tests PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
