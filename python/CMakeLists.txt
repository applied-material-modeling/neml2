# ----------------------------------------------------------------------------
# Python3 development headers and libraries
# ----------------------------------------------------------------------------
find_package(Python3 COMPONENTS Development.Module)

# ----------------------------------------------------------------------------
# Torch
# ----------------------------------------------------------------------------
get_target_property(torch_LINK_DIR torch::python INTERFACE_LINK_DIRECTORIES)

# ----------------------------------------------------------------------------
# Main interface library
# ----------------------------------------------------------------------------
add_custom_target(pyneml2)

# ----------------------------------------------------------------------------
# Macro for defining a submodule
# ----------------------------------------------------------------------------
macro(add_submodule mname malias)
  # sources
  file(GLOB_RECURSE msrcs CONFIGURE_DEPENDS src/csrc/${malias}/*.cxx)
  list(APPEND msrcs src/${malias}.cxx)

  # pybind11 module
  add_library(${mname} MODULE ${msrcs})
  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR}/python/src ${Python3_INCLUDE_DIRS})
  target_link_libraries(${mname} PUBLIC torch::python Python3::Module)
  set_target_properties(${mname} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY neml2
    OUTPUT_NAME ${malias}
    PREFIX ""
    SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
    CXX_VISIBILITY_PRESET hidden
  )
  if(DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable")
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/lib;${torch_LINK_DIR}")
  else()
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH}/lib;${INSTALL_REL_PATH}/../torch/lib")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET ${mname} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()

  if(DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "editable")
    # For an editable install, imports will be redirected to the source directory, so we need
    # to put the cpython libraries also in the source tree. All stubs (e.g. .pyi) will still be
    # installed to the site-packages, so LSP like pylance will still work correctly.
    install(TARGETS ${mname} LIBRARY DESTINATION ${NEML2_SOURCE_DIR}/python/neml2)
  else()
    install(TARGETS ${mname} LIBRARY DESTINATION .)
  endif()

  # link to the wrapper library
  add_dependencies(pyneml2 ${mname})
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
