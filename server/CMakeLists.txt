# ------------------------------------------------------------------------------
# WASP
# ------------------------------------------------------------------------------
find_package(WASP REQUIRED)

# ------------------------------------------------------------------------------
# neml2ls
# ------------------------------------------------------------------------------
add_executable(langserv
  src/main.cxx
  src/LanguageServer.cxx
)

target_include_directories(langserv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(langserv PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(langserv PRIVATE neml2 WASP)
set_target_properties(langserv PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS langserv COMPONENT libneml2-bin)
