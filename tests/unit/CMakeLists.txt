file(GLOB_RECURSE all_srcs CONFIGURE_DEPENDS *.cxx)
file(GLOB_RECURSE tensor_srcs CONFIGURE_DEPENDS tensors/*.cxx)

set(unit_srcs ${all_srcs})
list(REMOVE_ITEM unit_srcs ${tensor_srcs})

add_executable(unit_tests ${unit_srcs})
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(unit_tests PRIVATE testutils)
target_compile_options(unit_tests PRIVATE -Wall -Wextra -pedantic)

add_executable(tensor_tests ${tensor_srcs} main.cxx)
target_include_directories(tensor_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(tensor_tests PRIVATE testutils)
target_compile_options(tensor_tests PRIVATE -Wall -Wextra -pedantic)
