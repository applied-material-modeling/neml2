cmake_minimum_required(VERSION 3.26)
project(examples LANGUAGES CXX)
find_package(neml2 CONFIG REQUIRED)

# ----------------------------------------------------------------------------
# C++ examples
# ----------------------------------------------------------------------------
file(GLOB_RECURSE ex_srcs CONFIGURE_DEPENDS *.cxx)
set(ex_paths "")

foreach(ex_src ${ex_srcs})
  # example paranet directory and name
  get_filename_component(ex_src_dir ${ex_src} DIRECTORY)
  get_filename_component(ex_name ${ex_src} NAME_WE)
  cmake_path(RELATIVE_PATH ex_src BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE ex_src_relpath)
  cmake_path(RELATIVE_PATH ex_src_dir BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE ex_parent_relpath)

  # target name
  string(REPLACE "/" "_" ex_target ${ex_src_relpath})
  get_filename_component(ex_target ${ex_target} NAME_WE)

  # create target
  add_executable(${ex_target} ${ex_src})
  target_link_libraries(${ex_target} PRIVATE neml2::neml2)
  set_target_properties(${ex_target}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ex_parent_relpath}
    OUTPUT_NAME ${ex_name}
    DISABLE_PRECOMPILE_HEADERS ON
  )

  # add to list of examples
  list(APPEND ex_paths "${CMAKE_CURRENT_BINARY_DIR}/${ex_parent_relpath}/${ex_name}")
endforeach()

# -----------------------------------------------------------------------------
# Dump list of examples
# -----------------------------------------------------------------------------
set(FILE "${CMAKE_CURRENT_BINARY_DIR}/examples.txt")
file(WRITE ${FILE} "")
foreach(ex_path IN LISTS ex_paths)
  file(APPEND ${FILE} "${ex_path}\n")
endforeach()
