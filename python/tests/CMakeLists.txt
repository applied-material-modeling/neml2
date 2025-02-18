# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter)
execute_process(
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND ${Python_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/check_python_dep.py ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
)

# ----------------------------------------------------------------------------
# Install test resources
# ----------------------------------------------------------------------------
install(DIRECTORY .
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/neml2/python
  COMPONENT libneml2-python
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.i"
  PATTERN "*.pt"
  PATTERN "__pycache__" EXCLUDE
)
