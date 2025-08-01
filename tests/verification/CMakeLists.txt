file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cxx)
add_executable(verification_tests ${srcs})
target_link_libraries(verification_tests PRIVATE testutils)
target_compile_options(verification_tests PRIVATE -Wall -Wextra -pedantic)
