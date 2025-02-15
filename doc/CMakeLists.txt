if(NOT NEML2_PYBIND)
  message(WARNING "Documentation requires Python bindings (NEML2_PYBIND=ON), reverting NEML2_DOC to OFF")
  set(NEML2_DOC OFF CACHE BOOL "Build NEML2 documentation (html)" FORCE)
  return()
endif()

# ----------------------------------------------------------------------------
# Doxygen
# ----------------------------------------------------------------------------
find_package(Doxygen COMPONENTS dot)

if(NOT DOXYGEN_FOUND)
  include(DownloadDoxygen)

  if(doxygen_POPULATED)
    set(Doxygen_ROOT ${doxygen_SOURCE_DIR} CACHE PATH "Path to Doxygen installation" FORCE)
    find_package(Doxygen COMPONENTS dot)
  endif()
endif()

if(NOT DOXYGEN_FOUND)
  message(WARNING "Documentation requires Doxygen, but one cannot be located, reverting NEML2_DOC to OFF")
  set(NEML2_DOC OFF CACHE BOOL "Build NEML2 documentation (html)" FORCE)
  return()
endif()

get_target_property(DOXYGEN_EXECUTABLE Doxygen::doxygen IMPORTED_LOCATION)

# ----------------------------------------------------------------------------
# Doxygen html stylesheet
# ----------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  doxygen-awesome-css
  GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
  GIT_TAG v${DOXYGEN_AWESOME_VERSION}
)
FetchContent_MakeAvailable(doxygen-awesome-css)

# ----------------------------------------------------------------------------
# Python
# ----------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND ${Python_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/check_python_dep.py ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
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
  DEPENDS syntax
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc
  COMMAND ${NEML2_BINARY_DIR}/doc/syntax
  COMMAND ${Python_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/syntax_to_md.py syntax.yml content/syntax syntax_error.log
  VERBATIM
)

# ----------------------------------------------------------------------------
# Extract all Python API
# ----------------------------------------------------------------------------
add_custom_target(syntax_python
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
  COMMAND ${CMAKE_COMMAND} -E make_directory ${NEML2_BINARY_DIR}/doc/content/python
  COMMAND PYTHONPATH=. pybind11-stubgen -o ${NEML2_BINARY_DIR}/doc/content/python neml2
  COMMAND ${Python_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/fixup_pystub.py ${NEML2_BINARY_DIR}/doc/content/python/neml2
  VERBATIM
)
add_dependencies(syntax_python python_stub)

# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfileHTML "config/Doxyfile.in;config/HTML.in")
generate_doxyfile(${NEML2_BINARY_DIR}/doc/DoxyfilePython "config/Doxyfile.in;config/HTML.in;config/Python.in")
add_custom_target(html
  DEPENDS syntax_cpp syntax_python
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/doc
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfileHTML.sh
  COMMAND ${DOXYGEN_EXECUTABLE} -q DoxyfilePython.sh
  VERBATIM
)
