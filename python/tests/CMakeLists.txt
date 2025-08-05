# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
execute_process(
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/check_python_dep.py ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
)
