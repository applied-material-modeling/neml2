file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cxx)
add_executable(regression_tests ${srcs})
target_link_libraries(regression_tests PRIVATE testutils)
target_compile_options(regression_tests PRIVATE -Wall -Wextra -pedantic)
