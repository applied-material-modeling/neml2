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
target_link_libraries(base PUBLIC tensor nmhit::nmhit)

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

file(WRITE ${NEML2_BINARY_DIR}/version "v${PROJECT_VERSION}\n")

if(Git_FOUND)
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

# Whether dependencies are bundled with neml2
# torch is never installed as part of a standard NEML2 installation.
set(_torch_bundled OFF)
set(_nmhit_bundled OFF)
if(nmhit_CONTRIB OR NEML2_WHEEL)
  set(_nmhit_bundled ON)
endif()
if(NEML2_JSON)
  set(_nlohmann_json_bundled OFF)
  if(nlohmann_json_CONTRIB OR NEML2_WHEEL)
    set(_nlohmann_json_bundled ON)
  endif()
endif()
if(NEML2_CSV)
  set(_csvparser_bundled OFF)
  if(csvparser_CONTRIB OR NEML2_WHEEL)
    set(_csvparser_bundled ON)
  endif()
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

# Generates a per-dependency .pc file.
# Uses the bundled template (with relocatable ${prefix}) when BUNDLED is TRUE,
# otherwise the external template with absolute paths.
function(neml2_pkgconfig_generate_dep_pc)
  cmake_parse_arguments(ARG "" "OUTPUT_NAME;NAME;DESCRIPTION;VERSION;LIBS;CFLAGS;BUNDLED" "" ${ARGN})
  set(NEML2_DEP_PC_NAME        "${ARG_NAME}")
  set(NEML2_DEP_PC_DESCRIPTION "${ARG_DESCRIPTION}")
  set(NEML2_DEP_PC_VERSION     "${ARG_VERSION}")
  set(NEML2_DEP_PC_LIBS        "${ARG_LIBS}")
  set(NEML2_DEP_PC_CFLAGS      "${ARG_CFLAGS}")
  if(ARG_BUNDLED)
    configure_file(
      ${NEML2_SOURCE_DIR}/cmake/neml2-dep-bundled.pc.in
      ${NEML2_BINARY_DIR}/${ARG_OUTPUT_NAME}.pc
      @ONLY
    )
  else()
    configure_file(
      ${NEML2_SOURCE_DIR}/cmake/neml2-dep-external.pc.in
      ${NEML2_BINARY_DIR}/${ARG_OUTPUT_NAME}.pc
      @ONLY
    )
  endif()
endfunction()

# neml2-torch.pc (torch is always external)
get_filename_component(_torch_link_dir "${torch_LIBRARY}" DIRECTORY)
set(_torch_pc_libs "-L${_torch_link_dir} -lc10 -ltorch -ltorch_cpu")
if(TARGET torch::cuda)
  get_filename_component(_c10_cuda_link_dir "${c10_cuda_LIBRARY}" DIRECTORY)
  if(NOT _c10_cuda_link_dir STREQUAL _torch_link_dir)
    string(APPEND _torch_pc_libs " -L${_c10_cuda_link_dir}")
  endif()
  string(APPEND _torch_pc_libs " -lc10_cuda -ltorch_cuda -ltorch_cuda_linalg")
endif()
neml2_pkgconfig_generate_dep_pc(
  OUTPUT_NAME  neml2-torch
  NAME         "neml2-torch"
  DESCRIPTION  "LibTorch dependency for NEML2"
  VERSION      ""
  LIBS         "${_torch_pc_libs}"
  CFLAGS       "-I${torch_INCLUDE_DIR} -I${torch_csrc_INCLUDE_DIR}"
  BUNDLED      FALSE
)

# neml2-nmhit.pc
get_filename_component(_nmhit_lib_name "${nmhit_LIBRARY}" NAME_WE)
string(REGEX REPLACE "^lib" "" _nmhit_lib_name "${_nmhit_lib_name}")
if(_nmhit_bundled)
  set(_nmhit_pc_libs   "-L\${libdir} -l${_nmhit_lib_name}")
  set(_nmhit_pc_cflags "-I\${includedir}")
else()
  get_filename_component(_nmhit_lib_dir "${nmhit_LIBRARY}" DIRECTORY)
  set(_nmhit_pc_libs   "-L${_nmhit_lib_dir} -l${_nmhit_lib_name}")
  set(_nmhit_pc_include_dirs ${nmhit_INCLUDE_DIR})
  string(JOIN " -I" _nmhit_pc_cflags ${_nmhit_pc_include_dirs})
  set(_nmhit_pc_cflags "-I${_nmhit_pc_cflags}")
endif()
neml2_pkgconfig_generate_dep_pc(
  OUTPUT_NAME  neml2-nmhit
  NAME         "neml2-nmhit"
  DESCRIPTION  "nmhit dependency for NEML2"
  VERSION      ""
  LIBS         "${_nmhit_pc_libs}"
  CFLAGS       "${_nmhit_pc_cflags}"
  BUNDLED      ${_nmhit_bundled}
)

# neml2-nlohmann-json.pc (header-only; always generated, empty when NEML2_JSON is OFF)
if(NEML2_JSON AND _nlohmann_json_bundled)
  set(_json_pc_cflags "-I\${includedir}")
  set(_json_pc_bundled TRUE)
elseif(NEML2_JSON)
  set(_json_pc_cflags "-I${nlohmann_json_DIR}/include")
  set(_json_pc_bundled FALSE)
else()
  set(_json_pc_cflags "")
  set(_json_pc_bundled FALSE)
endif()
neml2_pkgconfig_generate_dep_pc(
  OUTPUT_NAME  neml2-nlohmann-json
  NAME         "neml2-nlohmann-json"
  DESCRIPTION  "nlohmann/json dependency for NEML2"
  VERSION      ""
  LIBS         ""
  CFLAGS       "${_json_pc_cflags}"
  BUNDLED      ${_json_pc_bundled}
)

# neml2-csvparser.pc (header-only; always generated, empty when NEML2_CSV is OFF)
if(NEML2_CSV AND _csvparser_bundled)
  set(_csv_pc_cflags "-I\${includedir}")
  set(_csv_pc_bundled TRUE)
elseif(NEML2_CSV)
  set(_csv_pc_cflags "-I${csvparser_INCLUDE_DIR}")
  set(_csv_pc_bundled FALSE)
else()
  set(_csv_pc_cflags "")
  set(_csv_pc_bundled FALSE)
endif()
neml2_pkgconfig_generate_dep_pc(
  OUTPUT_NAME  neml2-csvparser
  NAME         "neml2-csvparser"
  DESCRIPTION  "csvparser dependency for NEML2"
  VERSION      ""
  LIBS         ""
  CFLAGS       "${_csv_pc_cflags}"
  BUNDLED      ${_csv_pc_bundled}
)

# Build Requires: list for neml2.pc (only enabled features contribute flags)
set(NEML2_PKGCONFIG_REQUIRES "neml2-torch neml2-nmhit")
if(NEML2_JSON)
  string(APPEND NEML2_PKGCONFIG_REQUIRES " neml2-nlohmann-json")
endif()
if(NEML2_CSV)
  string(APPEND NEML2_PKGCONFIG_REQUIRES " neml2-csvparser")
endif()

set(NEML2_PKGCONFIG_SUFFIX "$<IF:$<CONFIG:Release>,,_$<CONFIG>>")

# neml2.pc (meta: pulls in core + all deps via Requires)
configure_file(
  ${NEML2_SOURCE_DIR}/cmake/neml2.pc.in
  ${NEML2_BINARY_DIR}/neml2.pc.in
  @ONLY
)
file(GENERATE
  OUTPUT ${NEML2_BINARY_DIR}/neml2${NEML2_PKGCONFIG_SUFFIX}.pc
  INPUT ${NEML2_BINARY_DIR}/neml2.pc.in
)

# neml2-core.pc (just NEML2's own libraries, no dependency flags)
configure_file(
  ${NEML2_SOURCE_DIR}/cmake/neml2-core.pc.in
  ${NEML2_BINARY_DIR}/neml2-core.pc.in
  @ONLY
)
file(GENERATE
  OUTPUT ${NEML2_BINARY_DIR}/neml2-core${NEML2_PKGCONFIG_SUFFIX}.pc
  INPUT ${NEML2_BINARY_DIR}/neml2-core.pc.in
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
    ${NEML2_BINARY_DIR}/neml2-core${NEML2_PKGCONFIG_SUFFIX}.pc
    ${NEML2_BINARY_DIR}/neml2-torch.pc
    ${NEML2_BINARY_DIR}/neml2-nmhit.pc
    ${NEML2_BINARY_DIR}/neml2-nlohmann-json.pc
    ${NEML2_BINARY_DIR}/neml2-csvparser.pc
    DESTINATION share/pkgconfig
    COMPONENT libneml2
  )
endif()
