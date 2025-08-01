file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cxx)
add_executable(dispatcher_tests ${srcs})
target_link_libraries(dispatcher_tests PRIVATE testutils)
target_compile_options(dispatcher_tests PRIVATE -Wall -Wextra -pedantic)
