# ----------------------------------------------------------------------------
# Torch
# ----------------------------------------------------------------------------
find_package(Torch)

if(NOT Torch_IS_PYTHON_PACKAGE)
  message(WARNING "Python bindings can only be built against a Torch python package. Although a Torch is found, it does not appear to come from Python site packages. Reverting NEML2_PYBIND to OFF.")
  set(NEML2_PYBIND OFF CACHE BOOL "Build Python bindings" FORCE)
  return()
endif()

# ----------------------------------------------------------------------------
# pybind11
# ----------------------------------------------------------------------------
include(DownloadPybind11)

# ----------------------------------------------------------------------------
# Macro for defining a submodule
# ----------------------------------------------------------------------------
macro(add_submodule mname msrcs)
  pybind11_add_module(${mname} MODULE ${msrcs})
  set_target_properties(${mname} PROPERTIES LIBRARY_OUTPUT_DIRECTORY neml2)

  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR})
  target_link_libraries(${mname} PRIVATE pybind11::headers)

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

add_submodule(reserved neml2/reserved.cxx)
target_link_libraries(reserved PRIVATE neml2_base Torch::PythonBinding)
set_target_properties(reserved PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

add_submodule(core neml2/core.cxx)
target_link_libraries(core PRIVATE neml2 Torch::PythonBinding)
set_target_properties(core PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

file(GLOB_RECURSE tensors_srcs CONFIGURE_DEPENDS neml2/tensors/*.cxx)
list(APPEND tensors_srcs neml2/tensors.cxx)
add_submodule(tensors "${tensors_srcs}")
target_link_libraries(tensors PRIVATE neml2_tensor Torch::PythonBinding)
set_target_properties(tensors PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

add_submodule(math neml2/math.cxx)
target_link_libraries(math PRIVATE neml2_tensor Torch::PythonBinding)
set_target_properties(math PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

add_submodule(crystallography neml2/crystallography.cxx)
target_link_libraries(crystallography PRIVATE neml2_tensor Torch::PythonBinding)
set_target_properties(crystallography PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

# ----------------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------------
configure_file(neml2/__init__.py ${NEML2_BINARY_DIR}/python/neml2/__init__.py COPYONLY)
configure_file(neml2/visualization.py ${NEML2_BINARY_DIR}/python/neml2/visualization.py COPYONLY)
configure_file(neml2/pyzag/__init__.py ${NEML2_BINARY_DIR}/python/neml2/pyzag/__init__.py COPYONLY)
configure_file(neml2/pyzag/interface.py ${NEML2_BINARY_DIR}/python/neml2/pyzag/interface.py COPYONLY)
configure_file(neml2/postprocessing/__init__.py ${NEML2_BINARY_DIR}/python/neml2/postprocessing/__init__.py COPYONLY)
configure_file(neml2/postprocessing/polefigure.py ${NEML2_BINARY_DIR}/python/neml2/postprocessing/polefigure.py COPYONLY)
configure_file(neml2/postprocessing/odf.py ${NEML2_BINARY_DIR}/python/neml2/postprocessing/odf.py COPYONLY)

install(FILES
  neml2/__init__.py
  neml2/visualization.py
  neml2/pyzag/__init__.py
  neml2/pyzag/interface.py
  neml2/postprocessing/__init__.py
  neml2/postprocessing/polefigure.py
  neml2/postprocessing/odf.py
  ${NEML2_BINARY_DIR}/python/neml2/__init__.pyi
  ${NEML2_BINARY_DIR}/python/neml2/core.pyi
  ${NEML2_BINARY_DIR}/python/neml2/reserved.pyi
  ${NEML2_BINARY_DIR}/python/neml2/math.pyi
  ${NEML2_BINARY_DIR}/python/neml2/crystallography.pyi
  ${NEML2_BINARY_DIR}/python/neml2/tensors.pyi
  DESTINATION .
  COMPONENT libneml2-python
)

# ----------------------------------------------------------------------------
# pyzag subpackage
# ----------------------------------------------------------------------------
configure_file(neml2/pyzag/__init__.py ${NEML2_BINARY_DIR}/python/neml2/pyzag/__init__.py COPYONLY)
configure_file(neml2/pyzag/interface.py ${NEML2_BINARY_DIR}/python/neml2/pyzag/interface.py COPYONLY)

install(DIRECTORY neml2/pyzag DESTINATION . COMPONENT libneml2-python)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(NEML2_TESTS)
  add_subdirectory(tests)
endif()
