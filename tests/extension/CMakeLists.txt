add_library(extension SHARED FooModel.cxx)
target_include_directories(extension PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(extension PRIVATE neml2)
target_compile_options(extension PRIVATE -Wall -Wextra -pedantic)
set_target_properties(extension
  PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set(EXTENSION_LIB ${CMAKE_CURRENT_SOURCE_DIR}/libextension${CMAKE_SHARED_LIBRARY_SUFFIX})
configure_file(FooModel.i.in ${CMAKE_CURRENT_SOURCE_DIR}/FooModel.i)
