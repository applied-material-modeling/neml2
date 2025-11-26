if(NOT NEML2_PYBIND)
  message(WARNING "Documentation generation requires Python bindings to be built. Reverting NEML2_DOC to OFF.")
  set(NEML2_DOC OFF CACHE BOOL "Build NEML2 documentation (html)" FORCE)
  return()
endif()

# ----------------------------------------------------------------------------
# Python
# ----------------------------------------------------------------------------
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
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/scripts/syntax_to_md.py syntax.yml content/syntax syntax_error.log
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
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/scripts/fixup_pystub.py ${NEML2_BINARY_DIR}/doc/content/python/neml2
  VERBATIM
)
add_dependencies(syntax_python pyneml2)

# ----------------------------------------------------------------------------
# Copy documentation content to build directory
# This also triggers a reconfigure if any doc content changes
# ----------------------------------------------------------------------------
file(GLOB_RECURSE DOC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/content/*")
foreach(DOC ${DOC_FILES})
  cmake_path(RELATIVE_PATH DOC BASE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/content OUTPUT_VARIABLE DOC_REL)
  get_filename_component(DOC_EXT ${DOC} LAST_EXT)
  if(DOC_EXT STREQUAL ".md")
    configure_file(${DOC} "${NEML2_BINARY_DIR}/doc/content/${DOC_REL}.in" COPYONLY)
  else()
    configure_file(${DOC} "${NEML2_BINARY_DIR}/doc/content/${DOC_REL}" COPYONLY)
  endif()
endforeach()

# ----------------------------------------------------------------------------
# Pre-process markdown files
# ----------------------------------------------------------------------------
message(STATUS "Pre-processing markdown")
configure_file(${CMAKE_CURRENT_LIST_DIR}/config/DoxygenLayout.xml ${NEML2_BINARY_DIR}/doc/config/DoxygenLayout.xml COPYONLY)
execute_process(
  COMMAND ${Python3_EXECUTABLE} scripts/update_content.py content ${NEML2_BINARY_DIR}/doc/content
  COMMAND ${Python3_EXECUTABLE} scripts/preprocess.py ${NEML2_BINARY_DIR}/doc/config/DoxygenLayout.xml ${NEML2_BINARY_DIR}/doc/content
  COMMAND_ERROR_IS_FATAL ANY
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

# ----------------------------------------------------------------------------
# Build and run C++ tutorials
# ----------------------------------------------------------------------------
file(GLOB_RECURSE TUTORIAL_SOURCES ${NEML2_BINARY_DIR}/doc/content/tutorials/*.cxx)
add_custom_target(tutorials_cpp)

foreach(TUTORIAL_SOURCE ${TUTORIAL_SOURCES})
  get_filename_component(TUTORIAL_SOURCE_DIR ${TUTORIAL_SOURCE} DIRECTORY)
  get_filename_component(TUTORIAL_NAME ${TUTORIAL_SOURCE} NAME_WE)
  cmake_path(RELATIVE_PATH TUTORIAL_SOURCE BASE_DIRECTORY ${NEML2_BINARY_DIR}/doc/content/tutorials OUTPUT_VARIABLE TUTORIAL_SOURCE_RELPATH)
  string(REPLACE "/" "_" TUTORIAL_TARGET ${TUTORIAL_SOURCE_RELPATH})
  get_filename_component(TUTORIAL_TARGET ${TUTORIAL_TARGET} NAME_WE)
  add_executable(${TUTORIAL_TARGET} ${TUTORIAL_SOURCE})
  target_link_libraries(${TUTORIAL_TARGET} PRIVATE neml2)
  set_target_properties(${TUTORIAL_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TUTORIAL_SOURCE_DIR})
  set_target_properties(${TUTORIAL_TARGET} PROPERTIES OUTPUT_NAME ${TUTORIAL_NAME})

  # Capture the output of the tutorial
  add_custom_target("${TUTORIAL_TARGET}_run"
    COMMENT "Running C++ tutorial ${TUTORIAL_TARGET}"
    DEPENDS ${TUTORIAL_TARGET}
    COMMAND ${TUTORIAL_SOURCE_DIR}/${TUTORIAL_NAME} > ${TUTORIAL_SOURCE_DIR}/${TUTORIAL_NAME}.out
    WORKING_DIRECTORY ${TUTORIAL_SOURCE_DIR}
    VERBATIM
  )
  add_dependencies(tutorials_cpp ${TUTORIAL_TARGET}_run)
endforeach()

# ----------------------------------------------------------------------------
# Run Python tutorials
# ----------------------------------------------------------------------------
file(GLOB_RECURSE TUTORIAL_SOURCES ${NEML2_BINARY_DIR}/doc/content/tutorials/*.py)
add_custom_target(tutorials_python)

foreach(TUTORIAL_SOURCE ${TUTORIAL_SOURCES})
  get_filename_component(TUTORIAL_SOURCE_DIR ${TUTORIAL_SOURCE} DIRECTORY)
  get_filename_component(TUTORIAL_NAME ${TUTORIAL_SOURCE} NAME_WE)
  cmake_path(RELATIVE_PATH TUTORIAL_SOURCE BASE_DIRECTORY ${NEML2_BINARY_DIR}/doc/content/tutorials OUTPUT_VARIABLE TUTORIAL_SOURCE_RELPATH)
  string(REPLACE "/" "_" TUTORIAL_TARGET ${TUTORIAL_SOURCE_RELPATH})
  get_filename_component(TUTORIAL_TARGET ${TUTORIAL_TARGET} NAME_WE)

  # Capture the output of the tutorial
  add_custom_target(${TUTORIAL_TARGET}
    COMMENT "Running Python tutorial ${TUTORIAL_TARGET}"
    DEPENDS "syntax_python;tutorials_cpp"
    COMMAND PYTHONPATH=${NEML2_BINARY_DIR}/python ${Python3_EXECUTABLE} ${TUTORIAL_SOURCE} > ${TUTORIAL_SOURCE_DIR}/${TUTORIAL_NAME}.out
    WORKING_DIRECTORY ${TUTORIAL_SOURCE_DIR}
    VERBATIM
  )
  add_dependencies(tutorials_python ${TUTORIAL_TARGET})
endforeach()

# ----------------------------------------------------------------------------
# Post-process markdown files
# ----------------------------------------------------------------------------
add_custom_target(tutorials_post
  COMMENT "Post-processing markdown"
  DEPENDS "tutorials_cpp;tutorials_python"
  COMMAND ${Python3_EXECUTABLE} scripts/postprocess.py ${NEML2_BINARY_DIR}/doc/content
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfileHTML "config/Doxyfile.in;config/HTML.in")
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfilePython "config/Doxyfile.in;config/HTML.in;config/Python.in")
add_custom_target(html
  COMMENT "Generating HTML documentation"
  DEPENDS "syntax_cpp;syntax_python;tutorials_post"
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfileHTML.sh
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfilePython.sh
  VERBATIM
)
