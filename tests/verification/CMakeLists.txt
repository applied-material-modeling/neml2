file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cxx)
add_executable(verification_tests ${srcs})
target_link_libraries(verification_tests PRIVATE testutils Catch2::Catch2)
target_compile_options(verification_tests PRIVATE -Wall -Wextra -pedantic)
set_target_properties(verification_tests PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
