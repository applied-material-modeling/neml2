message(STATUS "Configuring pybind11")
FetchContent_MakeAvailable(pybind11)

if(NOT Torch_PYTHON_BINDING)
  message(FATAL_ERROR "Could not find the libTorch Python binding")
endif()

# Macro for defining a submodule with given source files
macro(add_submodule mname msrcs)
  pybind11_add_module(${mname} MODULE ${msrcs})
  set_target_properties(${mname} PROPERTIES LIBRARY_OUTPUT_DIRECTORY neml2)
  set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${EXEC_DIR}/lib;${Torch_LINK_DIRECTORIES}")
  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR})
  target_link_libraries(${mname} PRIVATE pybind11::headers)
  target_link_libraries(${mname} PUBLIC neml2 ${Torch_PYTHON_BINDING})
  install(TARGETS ${mname} LIBRARY DESTINATION .)
endmacro()

# Actually define the submodules
add_submodule(base neml2/base.cxx)
add_submodule(math neml2/math.cxx)
file(GLOB_RECURSE tensors_srcs neml2/tensors/*.cxx)
add_submodule(tensors "${tensors_srcs}")

# Extract stub
add_custom_target(python-stub ALL
  DEPENDS base math tensors
  WORKING_DIRECTORY ${NEML2_BINARY_DIR}/python
  COMMAND PYTHONPATH=. pybind11-stubgen -o . neml2
  VERBATIM
)

# Artifacts
configure_file(neml2/__init__.py ${NEML2_BINARY_DIR}/python/neml2/__init__.py COPYONLY)
configure_file(neml2/visualization.py ${NEML2_BINARY_DIR}/python/neml2/visualization.py COPYONLY)
install(FILES
  neml2/__init__.py
  neml2/visualization.py
  ${NEML2_BINARY_DIR}/python/neml2/__init__.pyi
  ${NEML2_BINARY_DIR}/python/neml2/base.pyi
  ${NEML2_BINARY_DIR}/python/neml2/math.pyi
  ${NEML2_BINARY_DIR}/python/neml2/tensors.pyi
  DESTINATION .
)

# pyzag subpackage
configure_file(neml2/pyzag/__init__.py ${NEML2_BINARY_DIR}/python/neml2/pyzag/__init__.py COPYONLY)
configure_file(neml2/pyzag/interface.py ${NEML2_BINARY_DIR}/python/neml2/pyzag/interface.py COPYONLY)
install(DIRECTORY neml2/pyzag DESTINATION .)

# Tests
if(NEML2_TESTS)
  add_subdirectory(tests)
endif()
