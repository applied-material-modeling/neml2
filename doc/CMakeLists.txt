# ----------------------------------------------------------------------------
# Python
# ----------------------------------------------------------------------------
find_package(Python3 REQUIRED)
execute_process(
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/check_python_dep.py ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
)

# ----------------------------------------------------------------------------
# Macro for generating and configuring Doxyfile
# ----------------------------------------------------------------------------
macro(generate_doxyfile output inputs)
  file(WRITE ${output}.in "")

  foreach(input ${inputs})
    file(READ ${input} _content)
    file(APPEND ${output}.in ${_content})
  endforeach()

  configure_file(${output}.in ${output}.sh)

  file(REMOVE ${output}.in)
endmacro()

# ----------------------------------------------------------------------------
# Extract all input file syntax
# ----------------------------------------------------------------------------
add_executable(syntax syntax.cxx)
target_link_libraries(syntax PRIVATE neml2)
add_custom_target(syntax_cpp
  COMMENT "Generating libneml2 syntax"
  DEPENDS syntax
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc
  COMMAND ${NEML2_BINARY_DIR}/doc/syntax
  COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/syntax_to_md.py syntax.yml content/syntax syntax_error.log
  VERBATIM
)

# ----------------------------------------------------------------------------
# Extract all Python API
# ----------------------------------------------------------------------------
add_custom_target(syntax_python
  COMMENT "Rescoping type hints namespace"
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
  COMMAND ${CMAKE_COMMAND} -E make_directory ${NEML2_BINARY_DIR}/doc/content/python
  COMMAND PYTHONPATH=. pybind11-stubgen -o ${NEML2_BINARY_DIR}/doc/content/python neml2
  COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/fixup_pystub.py ${NEML2_BINARY_DIR}/doc/content/python/neml2
  VERBATIM
)
add_dependencies(syntax_python python_stub)

# ----------------------------------------------------------------------------
# Extract tutorial sources
# ----------------------------------------------------------------------------
message(STATUS "Extracting tutorial sources")
add_custom_target(tutorials)
file(REMOVE_RECURSE ${NEML2_BINARY_DIR}/doc/content/tutorials)
file(COPY tutorials DESTINATION ${NEML2_BINARY_DIR}/doc/content)
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/extract_tutorial_sources.py ${NEML2_BINARY_DIR}/doc/content/tutorials
  COMMAND_ERROR_IS_FATAL ANY
)

# ----------------------------------------------------------------------------
# Build and run C++ tutorials
# ----------------------------------------------------------------------------
file(GLOB_RECURSE TUTORIAL_SOURCES CONFIGURE_DEPENDS ${NEML2_BINARY_DIR}/doc/content/tutorials/*.cxx)

foreach(TUTORIAL_SOURCE ${TUTORIAL_SOURCES})
  get_filename_component(TUTORIAL_SOURCE_DIR ${TUTORIAL_SOURCE} DIRECTORY)
  get_filename_component(TUTORIAL_NAME ${TUTORIAL_SOURCE} NAME_WE)
  cmake_path(RELATIVE_PATH TUTORIAL_SOURCE BASE_DIRECTORY ${NEML2_BINARY_DIR}/doc/content OUTPUT_VARIABLE TUTORIAL_SOURCE_RELPATH)
  string(REPLACE "/" "_" TUTORIAL_SOURCE_TARGET ${TUTORIAL_SOURCE_RELPATH})
  string(APPEND TUTORIAL_SOURCE_TARGET "_cpp")
  add_executable(${TUTORIAL_SOURCE_TARGET} ${TUTORIAL_SOURCE})
  target_link_libraries(${TUTORIAL_SOURCE_TARGET} PRIVATE neml2)
  set_target_properties(${TUTORIAL_SOURCE_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TUTORIAL_SOURCE_DIR})
  set_target_properties(${TUTORIAL_SOURCE_TARGET} PROPERTIES OUTPUT_NAME ${TUTORIAL_NAME})
  add_dependencies(tutorials ${TUTORIAL_SOURCE_TARGET})

  # Capture the output of the tutorial
  add_custom_command(
    COMMENT "Running tutorial ${TUTORIAL_SOURCE_TARGET}"
    TARGET ${TUTORIAL_SOURCE_TARGET} POST_BUILD
    COMMAND ${TUTORIAL_SOURCE_DIR}/${TUTORIAL_NAME} > ${TUTORIAL_SOURCE_DIR}/${TUTORIAL_NAME}.out
    WORKING_DIRECTORY ${TUTORIAL_SOURCE_DIR}
    VERBATIM
  )
endforeach()

# ----------------------------------------------------------------------------
# Run Python tutorials
# ----------------------------------------------------------------------------
file(GLOB_RECURSE TUTORIAL_SOURCES CONFIGURE_DEPENDS ${NEML2_BINARY_DIR}/doc/content/tutorials/*.py)

foreach(TUTORIAL_SOURCE ${TUTORIAL_SOURCES})
  get_filename_component(TUTORIAL_SOURCE_DIR ${TUTORIAL_SOURCE} DIRECTORY)
  get_filename_component(TUTORIAL_NAME ${TUTORIAL_SOURCE} NAME_WE)
  cmake_path(RELATIVE_PATH TUTORIAL_SOURCE BASE_DIRECTORY ${NEML2_BINARY_DIR}/doc/content OUTPUT_VARIABLE TUTORIAL_SOURCE_RELPATH)
  string(REPLACE "/" "_" TUTORIAL_SOURCE_TARGET ${TUTORIAL_SOURCE_RELPATH})
  string(APPEND TUTORIAL_SOURCE_TARGET "_python")
  add_custom_target(${TUTORIAL_SOURCE_TARGET}
    DEPENDS syntax_python
    COMMENT "Running tutorial ${TUTORIAL_SOURCE_TARGET}"
    COMMAND PYTHONPATH=${NEML2_BINARY_DIR}/python ${Python3_EXECUTABLE} ${TUTORIAL_SOURCE} > ${TUTORIAL_SOURCE_DIR}/${TUTORIAL_NAME}.out
    WORKING_DIRECTORY ${TUTORIAL_SOURCE_DIR}
    VERBATIM
  )
  add_dependencies(tutorials ${TUTORIAL_SOURCE_TARGET})
endforeach()

# ----------------------------------------------------------------------------
# Postprocess tutorial markdown
# ----------------------------------------------------------------------------
add_custom_target(tutorials_md
  COMMENT "Postprocessing tutorial markdown"
  DEPENDS tutorials
  COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/substitute_tutorial_output.py ${NEML2_BINARY_DIR}/doc/content/tutorials
  COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/apply_layout.py ${NEML2_SOURCE_DIR}/doc/config/DoxygenLayout.xml ${NEML2_BINARY_DIR}/doc/content/tutorials
)

# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfileHTML "config/Doxyfile.in;config/HTML.in")
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfilePython "config/Doxyfile.in;config/HTML.in;config/Python.in")
add_custom_target(html
  COMMENT "Generating HTML documentation"
  DEPENDS syntax_cpp syntax_python tutorials_md
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfileHTML.sh
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfilePython.sh
  VERBATIM
)
