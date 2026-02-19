# ----------------------------------------------------------------------------
# Helper function to add a submodule
# ----------------------------------------------------------------------------
function(neml2_add_submodule mname TYPE mdir)
  file(GLOB_RECURSE msrcs CONFIGURE_DEPENDS ${mdir}/*.cxx)
  file(GLOB_RECURSE mheaders CONFIGURE_DEPENDS ${NEML2_SOURCE_DIR}/include/neml2/${mdir}/*.h)
  add_library(${mname} ${TYPE} ${msrcs})

  target_sources(${mname}
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
    ${NEML2_SOURCE_DIR}/include
    ${NEML2_BINARY_DIR}/include
    FILES
    ${mheaders}
    ${NEML2_BINARY_DIR}/include/neml2/config.h
  )

  set_target_properties(${mname} PROPERTIES OUTPUT_NAME "neml2_${mname}$<IF:$<CONFIG:Release>,,_$<CONFIG>>")

  if(NEML2_WHEEL AND NOT SKBUILD_STATE STREQUAL "editable")
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH};${INSTALL_REL_PATH}/../../torch/lib")
  else()
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH};${torch_LINK_DIR}")
  endif()

  target_compile_options(${mname} PRIVATE -Wall -Wextra -pedantic)

  # Add library to the interface neml2 target
  target_link_libraries(neml2 PUBLIC ${mname})

  install(TARGETS ${mname}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    COMPONENT libneml2
  )
  if(NOT SKBUILD_STATE STREQUAL "editable")
    install(TARGETS ${mname}
      EXPORT neml2targets
      FILE_SET HEADERS DESTINATION include
      COMPONENT libneml2
    )
  endif()
endfunction()

# ----------------------------------------------------------------------------
# Submodules
# ----------------------------------------------------------------------------
# libneml2
# This is a wrapper of all neml2 library components.
add_library(neml2 SHARED "neml2.cxx")
target_sources(neml2
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS
  ${NEML2_SOURCE_DIR}/include
  ${NEML2_BINARY_DIR}/include
  FILES
  ${NEML2_SOURCE_DIR}/include/neml2/neml2.h
  ${NEML2_BINARY_DIR}/include/neml2/config.h
)
set_target_properties(neml2 PROPERTIES OUTPUT_NAME "neml2$<IF:$<CONFIG:Release>,,_$<CONFIG>>")
if(NEML2_WHEEL AND NOT SKBUILD_STATE STREQUAL "editable")
  set_target_properties(neml2 PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH};${INSTALL_REL_PATH}/../../torch/lib")
else()
  set_target_properties(neml2 PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH};${torch_LINK_DIR}")
endif()
target_compile_options(neml2 PRIVATE -Wall -Wextra -pedantic)

# libneml2_misc
neml2_add_submodule(misc SHARED misc)
target_link_libraries(misc PUBLIC torch::core)
target_link_options(misc PRIVATE ${CMAKE_CXX_LINK_WHAT_YOU_USE_FLAG})

if(TARGET torch::cuda)
  target_link_libraries(misc PUBLIC torch::cuda)
endif()

if(NEML2_JSON)
  target_link_libraries(misc PUBLIC nlohmann_json::nlohmann_json)
endif()

if(NEML2_CSV)
  target_link_libraries(misc PUBLIC csv)
endif()

if(NEML2_PCH)
  target_precompile_headers(misc PUBLIC
    <neml2/misc/types.h>
    <neml2/misc/assertions.h>
  )
endif()

# libneml2_tensor
neml2_add_submodule(tensor SHARED tensors)
target_link_libraries(tensor PUBLIC misc)

if(NEML2_PCH)
  target_precompile_headers(tensor PUBLIC
    <neml2/tensors/TensorBase.h>
    <neml2/tensors/TensorBaseImpl.h>
    <neml2/tensors/PrimitiveTensor.h>
    <neml2/tensors/assertions.h>
    <neml2/tensors/shape_utils.h>
  )
endif()

# libneml2_base
neml2_add_submodule(base SHARED base)
target_link_libraries(base PUBLIC tensor hit)

if(NEML2_PCH)
  target_precompile_headers(base PUBLIC
    <neml2/base/Registry.h>
    <neml2/base/Factory.h>
    <neml2/base/Option.h>
    <neml2/base/TensorName.h>
  )
endif()

# libneml2_user_tensor
neml2_add_submodule(user_tensor SHARED user_tensors)
target_link_libraries(user_tensor PUBLIC tensor base)

# libneml2_equation_system
neml2_add_submodule(equation_system SHARED equation_systems)
target_link_libraries(equation_system PUBLIC tensor base)

# libneml2_solver
neml2_add_submodule(solver SHARED solvers)
target_link_libraries(solver PUBLIC equation_system)

# libneml2_model
neml2_add_submodule(model SHARED models)
target_link_libraries(model PUBLIC solver tensor base)

if(NEML2_PCH)
  target_precompile_headers(model PUBLIC
    <neml2/models/Model.h>
  )
endif()

# libneml2_driver
neml2_add_submodule(driver SHARED drivers)
target_link_libraries(driver PUBLIC model tensor base)

# libneml2_dispatcher
if(NEML2_WORK_DISPATCHER)
  neml2_add_submodule(dispatcher SHARED dispatchers)
  target_link_libraries(dispatcher PUBLIC tensor model MPI::MPI_CXX)
  target_link_libraries(driver PUBLIC dispatcher)
endif()

configure_file(
  ${NEML2_SOURCE_DIR}/include/neml2/config.h.in
  ${NEML2_BINARY_DIR}/include/neml2/config.h
)

# ----------------------------------------------------------------------------
# Version number and hash
# ----------------------------------------------------------------------------
find_package(Git)

if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --abbrev=0
    WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
    OUTPUT_VARIABLE NEML2_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  file(WRITE ${NEML2_BINARY_DIR}/version "${NEML2_VERSION}\n")

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
    OUTPUT_VARIABLE NEML2_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  file(WRITE ${NEML2_BINARY_DIR}/hash "${NEML2_HASH}\n")

  install(FILES
    ${NEML2_BINARY_DIR}/version
    ${NEML2_BINARY_DIR}/hash
    DESTINATION .
    COMPONENT libneml2
  )
endif()

# ----------------------------------------------------------------------------
# Export targets
# ----------------------------------------------------------------------------
install(TARGETS neml2
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  COMPONENT libneml2
)
if(NOT SKBUILD_STATE STREQUAL "editable")
  install(TARGETS neml2
    EXPORT neml2targets
    FILE_SET HEADERS DESTINATION include
    COMPONENT libneml2
  )
  install(EXPORT neml2targets NAMESPACE neml2:: DESTINATION share/cmake/neml2 COMPONENT libneml2)
endif()

# cmake config
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${NEML2_SOURCE_DIR}/cmake/neml2Config.cmake.in
  ${NEML2_BINARY_DIR}/neml2Config.cmake
  INSTALL_DESTINATION share/cmake/neml2
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
  ${NEML2_BINARY_DIR}/neml2ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# pkgconfig .pc file
if(IS_ABSOLUTE "${INSTALL_LIBDIR}")
  set(NEML2_PKGCONFIG_LIBDIR "${INSTALL_LIBDIR}")
else()
  set(NEML2_PKGCONFIG_LIBDIR "\${exec_prefix}/${INSTALL_LIBDIR}")
endif()
# Keep the pkg-config file relocatable so cmake --install --prefix works.
# neml2.pc is installed to <prefix>/share/pkgconfig, so prefix is two levels up.
set(NEML2_PKGCONFIG_PREFIX "\${pcfiledir}/../..")

set(NEML2_PKGCONFIG_LIB_TARGETS neml2)
get_target_property(_neml2_interface_libs neml2 INTERFACE_LINK_LIBRARIES)
foreach(_lib IN LISTS _neml2_interface_libs)
  if(TARGET "${_lib}")
    list(APPEND NEML2_PKGCONFIG_LIB_TARGETS "${_lib}")
  endif()
endforeach()
list(REMOVE_DUPLICATES NEML2_PKGCONFIG_LIB_TARGETS)

set(NEML2_PKGCONFIG_LIBS "-L\${libdir}")
foreach(_lib_target IN LISTS NEML2_PKGCONFIG_LIB_TARGETS)
  string(APPEND NEML2_PKGCONFIG_LIBS " -l$<TARGET_FILE_BASE_NAME:${_lib_target}>")
endforeach()

# Helper to append include and link flags for dependencies.
function(neml2_pkgconfig_add_dep_include _include_dir _bundled)
  if(NOT "${_include_dir}" STREQUAL "" AND NOT "${_include_dir}" MATCHES "-NOTFOUND$")
    if(_bundled)
      list(APPEND NEML2_PKGCONFIG_DEP_INCLUDE_DIRS "\${includedir}")
    else()
      list(APPEND NEML2_PKGCONFIG_DEP_INCLUDE_DIRS "${_include_dir}")
    endif()
    set(NEML2_PKGCONFIG_DEP_INCLUDE_DIRS "${NEML2_PKGCONFIG_DEP_INCLUDE_DIRS}" PARENT_SCOPE)
  endif()
endfunction()

function(neml2_pkgconfig_add_dep_library _library_path _bundled)
  if(NOT "${_library_path}" STREQUAL "" AND NOT "${_library_path}" MATCHES "-NOTFOUND$")
    get_filename_component(_dep_lib_dir "${_library_path}" DIRECTORY)
    get_filename_component(_dep_lib_name "${_library_path}" NAME_WE)
    string(REGEX REPLACE "^lib" "" _dep_lib_name "${_dep_lib_name}")
    if(_bundled)
      list(APPEND NEML2_PKGCONFIG_DEP_LINK_DIRS "\${libdir}")
    else()
      list(APPEND NEML2_PKGCONFIG_DEP_LINK_DIRS "${_dep_lib_dir}")
    endif()
    list(APPEND NEML2_PKGCONFIG_DEP_LIB_FLAGS "-l${_dep_lib_name}")
    set(NEML2_PKGCONFIG_DEP_LINK_DIRS "${NEML2_PKGCONFIG_DEP_LINK_DIRS}" PARENT_SCOPE)
    set(NEML2_PKGCONFIG_DEP_LIB_FLAGS "${NEML2_PKGCONFIG_DEP_LIB_FLAGS}" PARENT_SCOPE)
  endif()
endfunction()

set(NEML2_PKGCONFIG_DEP_INCLUDE_DIRS
  "${torch_INCLUDE_DIR}"
  "${torch_csrc_INCLUDE_DIR}"
)
set(NEML2_PKGCONFIG_DEP_LINK_DIRS)
set(NEML2_PKGCONFIG_DEP_LIB_FLAGS)

# torch is never installed as part of a standard NEML2 installation.
neml2_pkgconfig_add_dep_library("${c10_LIBRARY}" OFF)
neml2_pkgconfig_add_dep_library("${torch_LIBRARY}" OFF)
neml2_pkgconfig_add_dep_library("${torch_cpu_LIBRARY}" OFF)
if(TARGET torch::cuda)
  neml2_pkgconfig_add_dep_library("${c10_cuda_LIBRARY}" OFF)
  neml2_pkgconfig_add_dep_library("${torch_cuda_LIBRARY}" OFF)
  neml2_pkgconfig_add_dep_library("${torch_cuda_linalg_LIBRARY}" OFF)
endif()

set(_hit_bundled OFF)
if(hit_CONTRIB OR NEML2_WHEEL)
  set(_hit_bundled ON)
endif()
neml2_pkgconfig_add_dep_include("${hit_INCLUDE_DIR}" ${_hit_bundled})
neml2_pkgconfig_add_dep_library("${hit_LIBRARY}" ${_hit_bundled})

set(_wasp_bundled OFF)
if(wasp_CONTRIB OR NEML2_WHEEL)
  set(_wasp_bundled ON)
endif()
neml2_pkgconfig_add_dep_include("${wasp_core_INCLUDE_DIR}" ${_wasp_bundled})
neml2_pkgconfig_add_dep_include("${wasp_hit_INCLUDE_DIR}" ${_wasp_bundled})
neml2_pkgconfig_add_dep_library("${wasp_core_LIBRARY}" ${_wasp_bundled})
neml2_pkgconfig_add_dep_library("${wasp_hit_LIBRARY}" ${_wasp_bundled})

if(NEML2_JSON)
  set(_nlohmann_json_bundled OFF)
  if(nlohmann_json_CONTRIB OR NEML2_WHEEL)
    set(_nlohmann_json_bundled ON)
  endif()
  neml2_pkgconfig_add_dep_include("${nlohmann_json_DIR}/include" ${_nlohmann_json_bundled})
endif()

if(NEML2_CSV)
  set(_csvparser_bundled OFF)
  if(csvparser_CONTRIB OR NEML2_WHEEL)
    set(_csvparser_bundled ON)
  endif()
  neml2_pkgconfig_add_dep_include("${csvparser_INCLUDE_DIR}" ${_csvparser_bundled})
endif()

list(REMOVE_DUPLICATES NEML2_PKGCONFIG_DEP_INCLUDE_DIRS)
list(REMOVE_DUPLICATES NEML2_PKGCONFIG_DEP_LINK_DIRS)
list(REMOVE_DUPLICATES NEML2_PKGCONFIG_DEP_LIB_FLAGS)

foreach(_dep_link_dir IN LISTS NEML2_PKGCONFIG_DEP_LINK_DIRS)
  if(NOT "${_dep_link_dir}" STREQUAL "\${libdir}")
    string(APPEND NEML2_PKGCONFIG_LIBS " -L${_dep_link_dir}")
  endif()
endforeach()
foreach(_dep_lib_flag IN LISTS NEML2_PKGCONFIG_DEP_LIB_FLAGS)
  string(APPEND NEML2_PKGCONFIG_LIBS " ${_dep_lib_flag}")
endforeach()

set(NEML2_PKGCONFIG_CFLAGS "-I\${includedir}")
foreach(_dep_include_dir IN LISTS NEML2_PKGCONFIG_DEP_INCLUDE_DIRS)
  if(NOT "${_dep_include_dir}" STREQUAL "\${includedir}")
    string(APPEND NEML2_PKGCONFIG_CFLAGS " -I${_dep_include_dir}")
  endif()
endforeach()

configure_file(
  ${NEML2_SOURCE_DIR}/cmake/neml2.pc.in
  ${NEML2_BINARY_DIR}/neml2.pc.in
  @ONLY
)
set(NEML2_PKGCONFIG_SUFFIX "$<IF:$<CONFIG:Release>,,_$<CONFIG>>")
file(GENERATE
  OUTPUT ${NEML2_BINARY_DIR}/neml2${NEML2_PKGCONFIG_SUFFIX}.pc
  INPUT ${NEML2_BINARY_DIR}/neml2.pc.in
)

if(NOT SKBUILD_STATE STREQUAL "editable")
  install(
    FILES
    ${NEML2_BINARY_DIR}/neml2Config.cmake
    ${NEML2_BINARY_DIR}/neml2ConfigVersion.cmake
    DESTINATION share/cmake/neml2
    COMPONENT libneml2
  )
  install(
    FILES
    ${NEML2_BINARY_DIR}/neml2${NEML2_PKGCONFIG_SUFFIX}.pc
    DESTINATION share/pkgconfig
    COMPONENT libneml2
  )
endif()
