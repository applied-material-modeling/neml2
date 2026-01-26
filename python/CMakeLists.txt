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

if(NOT ${torch_LINK_DIR} STREQUAL ${Python3_SITEARCH}/torch/lib AND NOT ${torch_LINK_DIR} STREQUAL ${Python3_USER_SITE}/torch/lib)
  message(WARNING "Python bindings can only be built against a Torch python package. A Torch package was found at ${torch_LINK_DIR}. It does not appear to come from Python site packages. Reverting NEML2_PYBIND to OFF.")
  set(NEML2_PYBIND OFF CACHE BOOL "Build Python bindings" FORCE)
  return()
endif()

# ----------------------------------------------------------------------------
# Main interface library
# ----------------------------------------------------------------------------
add_custom_target(pyneml2_modules)
add_custom_target(pyneml2)

# ----------------------------------------------------------------------------
# Macro for defining a submodule
# ----------------------------------------------------------------------------
macro(add_submodule mname malias)
  # sources
  file(GLOB_RECURSE msrcs CONFIGURE_DEPENDS neml2/csrc/${malias}/*.cxx)
  list(APPEND msrcs neml2/${malias}.cxx)

  # pybind11 module
  add_library(${mname} MODULE ${msrcs})
  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR} ${Python3_INCLUDE_DIRS})
  target_link_libraries(${mname} PUBLIC torch::python Python3::Module)
  set_target_properties(${mname} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY neml2
    OUTPUT_NAME ${malias}
    PREFIX ""
    SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
    CXX_VISIBILITY_PRESET hidden
  )
  if(NEML2_WHEEL)
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/lib;${INSTALL_REL_PATH}/../torch/lib")
  else()
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/lib;${torch_LINK_DIR}")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET ${mname} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
  install(TARGETS ${mname} LIBRARY DESTINATION . COMPONENT libneml2-python)

  # link to the wrapper library
  add_dependencies(pyneml2_modules ${mname})

  # add a target to extract stub
  add_custom_target(${mname}_stub
    COMMENT "Generating Python type hints for neml2.${malias}"
    WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
    COMMAND PYTHONPATH=. ${Python3_EXECUTABLE} -m pybind11_stubgen -o . neml2.${malias}
    VERBATIM
  )
  add_dependencies(${mname}_stub pyneml2_modules)

  # add to the catch-all target
  add_dependencies(pyneml2 ${mname}_stub)
endmacro()

# ----------------------------------------------------------------------------
# Python submodules
# ----------------------------------------------------------------------------
add_submodule(pyreserved reserved)
target_link_libraries(pyreserved PRIVATE base)

add_submodule(pycore core)
target_link_libraries(pycore PUBLIC neml2)

add_submodule(pytensors tensors)
target_link_libraries(pytensors PUBLIC tensor)

add_submodule(pymath math)
target_link_libraries(pymath PUBLIC tensor)

add_submodule(pycrystallography crystallography)
target_link_libraries(pycrystallography PUBLIC tensor)

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
