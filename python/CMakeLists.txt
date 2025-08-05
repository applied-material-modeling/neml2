if(NOT TARGET Python3::Module)
  message(FATAL_ERROR "Python3 Development.Module component not found. Please ensure Python3 is installed with development headers.")
endif()

# ----------------------------------------------------------------------------
# Torch
# ----------------------------------------------------------------------------
if(NOT TARGET torch::python)
  message(WARNING "Torch Python bindings not found. If you built torch from source, you need to build it with the Python bindings enabled. Reverting NEML2_PYBIND to OFF.")
  set(NEML2_PYBIND OFF CACHE BOOL "Build Python bindings" FORCE)
  return()
endif()

get_target_property(torch_LINK_DIR torch::python INTERFACE_LINK_DIRECTORIES)

if(NOT ${torch_LINK_DIR} STREQUAL ${Python3_SITEARCH}/torch/lib)
  message(WARNING "Python bindings can only be built against a Torch python package. Although a Torch is found, it does not appear to come from Python site packages. Reverting NEML2_PYBIND to OFF.")
  set(NEML2_PYBIND OFF CACHE BOOL "Build Python bindings" FORCE)
  return()
endif()

# ----------------------------------------------------------------------------
# pybind11
# ----------------------------------------------------------------------------
execute_process(
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/check_python_dep.py ${CMAKE_CURRENT_SOURCE_DIR}/neml2/requirements.txt
)
set(pybind11_ROOT ${Python3_SITEARCH}/pybind11)
find_package(pybind11 CONFIG REQUIRED)

# ----------------------------------------------------------------------------
# Macro for defining a submodule
# ----------------------------------------------------------------------------
macro(add_submodule mname malias msrcs)
  pybind11_add_module(${mname} MODULE ${msrcs})
  set_target_properties(${mname} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY neml2
    OUTPUT_NAME ${malias}
  )

  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR})
  target_link_libraries(${mname} PRIVATE pybind11::headers torch::python)

  if(NEML2_WHEEL)
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH};${INSTALL_REL_PATH}/lib;${INSTALL_REL_PATH}/../torch/lib")
  else()
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH};${INSTALL_REL_PATH}/lib;${torch_LINK_DIR}")
  endif()

  install(TARGETS ${mname} LIBRARY DESTINATION . COMPONENT libneml2-python)

  # link to the wrapper library
  target_link_libraries(pyneml2 INTERFACE ${mname})

  # add to the stub generation
  add_dependencies(python_stub ${mname})
endmacro()

# ----------------------------------------------------------------------------
# Extract stub
# ----------------------------------------------------------------------------
add_custom_target(python_stub
  COMMENT "Generating neml2 type hints"
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
  COMMAND PYTHONPATH=. pybind11-stubgen -o . neml2
  VERBATIM
)

# ----------------------------------------------------------------------------
# Python submodules
# ----------------------------------------------------------------------------
add_library(pyneml2 INTERFACE)

add_submodule(reserved reserved neml2/reserved.cxx)
target_link_libraries(reserved PRIVATE base)

add_submodule(core core neml2/core.cxx)
target_link_libraries(core PRIVATE neml2)

file(GLOB_RECURSE tensors_srcs CONFIGURE_DEPENDS neml2/tensors/*.cxx)
list(APPEND tensors_srcs neml2/tensors.cxx)
add_submodule(pytensors tensors "${tensors_srcs}")
target_link_libraries(pytensors PRIVATE tensor)

add_submodule(pymath math neml2/math.cxx)
target_link_libraries(pymath PRIVATE tensor)

add_submodule(pycrystallography crystallography neml2/crystallography.cxx)
target_link_libraries(pycrystallography PRIVATE tensor)

# ----------------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------------
file(COPY neml2/ DESTINATION ${NEML2_BINARY_DIR}/python/neml2 FILES_MATCHING PATTERN "*.py")
install(DIRECTORY
  ${NEML2_BINARY_DIR}/python/neml2/
  DESTINATION .
  COMPONENT libneml2-python
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.pyi"
)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(NEML2_TESTS)
  execute_process(
    COMMAND_ERROR_IS_FATAL ANY
    COMMAND ${Python3_EXECUTABLE} ${NEML2_SOURCE_DIR}/scripts/check_python_dep.py ${CMAKE_CURRENT_SOURCE_DIR}/tests/requirements.txt
  )
  add_subdirectory(tests)
endif()
