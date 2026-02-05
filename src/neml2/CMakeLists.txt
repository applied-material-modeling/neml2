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

  if(NEML2_WHEEL)
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH};${INSTALL_REL_PATH}/../../torch/lib")
  else()
    set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${INSTALL_REL_PATH};${torch_LINK_DIR}")
  endif()

  target_compile_options(${mname} PRIVATE -Wall -Wextra -pedantic)

  # Add library to the interface neml2 target
  target_link_libraries(neml2 PUBLIC ${mname})

  install(TARGETS ${mname}
    EXPORT neml2targets
    LIBRARY COMPONENT libneml2
    FILE_SET HEADERS COMPONENT libneml2
  )

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
if(NEML2_WHEEL)
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

# feature definitions
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
  file(WRITE ${NEML2_BINARY_DIR}/version ${NEML2_VERSION})

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
    OUTPUT_VARIABLE NEML2_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  file(WRITE ${NEML2_BINARY_DIR}/hash ${NEML2_HASH})

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
  EXPORT neml2targets
  LIBRARY COMPONENT libneml2
  FILE_SET HEADERS COMPONENT libneml2
)
install(EXPORT neml2targets NAMESPACE neml2:: DESTINATION share/cmake/neml2 COMPONENT libneml2)

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

install(
  FILES
  ${NEML2_BINARY_DIR}/neml2Config.cmake
  ${NEML2_BINARY_DIR}/neml2ConfigVersion.cmake
  DESTINATION share/cmake/neml2
  COMPONENT libneml2
)
