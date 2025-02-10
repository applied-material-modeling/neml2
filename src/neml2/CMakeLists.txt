# ----------------------------------------------------------------------------
# Common options for all submodules
# ----------------------------------------------------------------------------
option(NEML2_CPU_PROFILER "Link against gperftools libprofiler to enable CPU profiling" OFF)
option(NEML2_WORK_DISPATCHER "Enable NEML2 work dispatcher" OFF)

if(NEML2_CPU_PROFILER)
  message(STATUS "Configuring gperftools")
  FetchContent_MakeAvailable(gperftools)
endif()

# ----------------------------------------------------------------------------
# WASP
# ----------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(wasp_BUILD_TYPE "Debug")
else()
  set(wasp_BUILD_TYPE "RelWithDebInfo")
endif()

message(STATUS "Configuring WASP")
set(wasp_BUILD_DIR ${NEML2_BINARY_DIR}/wasp/build)
set(wasp_INSTALL_DIR ${NEML2_BINARY_DIR}/wasp/install)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${wasp_BUILD_DIR})
execute_process(
  COMMAND ${CMAKE_COMMAND} \\
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \\
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \\
  -DVERSION=${WASP_VERSION} \\
  -DPREFIX=${FETCHCONTENT_BASE_DIR}/wasp \\
  -DBUILD_TYPE=${wasp_BUILD_TYPE} \\
  -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \\
  -DINSTALL_PREFIX=${wasp_INSTALL_DIR} \\
  ${NEML2_SOURCE_DIR}/cmake/wasp
  WORKING_DIRECTORY ${wasp_BUILD_DIR}
  OUTPUT_QUIET
  OUTPUT_FILE ${wasp_BUILD_DIR}/configure.log
  ERROR_QUIET
  ERROR_FILE ${wasp_BUILD_DIR}/configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "Installing WASP")
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${wasp_BUILD_DIR}
  OUTPUT_QUIET
  OUTPUT_FILE ${wasp_BUILD_DIR}/build.log
  ERROR_QUIET
  ERROR_FILE ${wasp_BUILD_DIR}/build.err
  COMMAND_ERROR_IS_FATAL ANY
)
install(DIRECTORY
  ${wasp_INSTALL_DIR}/include/waspcore
  ${wasp_INSTALL_DIR}/include/wasphit
  TYPE INCLUDE
)

# ----------------------------------------------------------------------------
# HIT
# ----------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(hit_BUILD_TYPE "Debug")
else()
  set(hit_BUILD_TYPE "RelWithDebInfo")
endif()

message(STATUS "Configuring HIT")
set(hit_BUILD_DIR ${NEML2_BINARY_DIR}/hit/build)
set(hit_INSTALL_DIR ${NEML2_BINARY_DIR}/hit/install)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${hit_BUILD_DIR})
execute_process(
  COMMAND ${CMAKE_COMMAND} \\
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \\
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \\
  -DVERSION=${HIT_VERSION} \\
  -Dwasp_INSTALL_DIR=${wasp_INSTALL_DIR} \\
  -DPREFIX=${FETCHCONTENT_BASE_DIR} \\
  -DBUILD_TYPE=${hit_BUILD_TYPE} \\
  -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \\
  -G${CMAKE_GENERATOR} \\
  ${NEML2_SOURCE_DIR}/cmake/hit
  WORKING_DIRECTORY ${hit_BUILD_DIR}
  OUTPUT_QUIET
  OUTPUT_FILE ${hit_BUILD_DIR}/configure.log
  ERROR_QUIET
  ERROR_FILE ${hit_BUILD_DIR}/configure.err
  COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "Installing HIT")
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${hit_BUILD_DIR}

  # OUTPUT_QUIET
  # OUTPUT_FILE ${hit_BUILD_DIR}/build.log
  # ERROR_QUIET
  # ERROR_FILE ${hit_BUILD_DIR}/build.err
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${hit_INSTALL_DIR})
execute_process(
  COMMAND ${CMAKE_COMMAND} --install ${hit_BUILD_DIR} --prefix ${hit_INSTALL_DIR}
  OUTPUT_QUIET
  OUTPUT_FILE ${hit_INSTALL_DIR}/install.log
  ERROR_QUIET
  ERROR_FILE ${hit_INSTALL_DIR}/install.err
  COMMAND_ERROR_IS_FATAL ANY
)
install(DIRECTORY
  ${hit_INSTALL_DIR}/include/hit
  TYPE INCLUDE
)

# ----------------------------------------------------------------------------
# Prepare for work dispatcher
# ----------------------------------------------------------------------------
if(NEML2_WORK_DISPATCHER)
  find_package(MPI REQUIRED)

  if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "MPI C++ compiler (required for work dispatcher) not found")
  endif()

  # TIMPI
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(timpi_BUILD_TYPE "dbg")
  else()
    set(timpi_BUILD_TYPE "opt")
  endif()

  message(STATUS "Configuring TIMPI")
  set(timpi_BUILD_DIR ${NEML2_BINARY_DIR}/timpi/build)
  set(timpi_INSTALL_DIR ${NEML2_BINARY_DIR}/timpi/install)
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${timpi_BUILD_DIR})
  execute_process(
    COMMAND ${CMAKE_COMMAND} \\
    -DCMAKE_C_COMPILER=${MPI_C_COMPILER} \\
    -DCMAKE_CXX_COMPILER=${MPI_CXX_COMPILER} \\
    -DVERSION=${TIMPI_VERSION} \\
    -DPREFIX=${FETCHCONTENT_BASE_DIR}/timpi \\
    -DBUILD_TYPE=${timpi_BUILD_TYPE} \\
    -DINSTALL_PREFIX=${timpi_INSTALL_DIR} \\
    ${NEML2_SOURCE_DIR}/cmake/timpi
    WORKING_DIRECTORY ${timpi_BUILD_DIR}
    OUTPUT_QUIET
    OUTPUT_FILE ${timpi_BUILD_DIR}/configure.log
    ERROR_QUIET
    ERROR_FILE ${timpi_BUILD_DIR}/configure.err
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${timpi_BUILD_DIR}
    WORKING_DIRECTORY ${timpi_BUILD_DIR}
    OUTPUT_QUIET
    OUTPUT_FILE ${timpi_BUILD_DIR}/build.log
    ERROR_QUIET
    ERROR_FILE ${timpi_BUILD_DIR}/build.err
    COMMAND_ERROR_IS_FATAL ANY
  )

  install(DIRECTORY
    ${timpi_INSTALL_DIR}/include/timpi
    TYPE INCLUDE
  )
endif()

# ----------------------------------------------------------------------------
# Macro for adding a submodule
# ----------------------------------------------------------------------------
macro(neml2_add_submodule mname TYPE mdir)
  list(APPEND NEML2_LIBS ${mname})

  file(GLOB_RECURSE msrcs ${NEML2_SOURCE_DIR}/src/neml2/${mdir}/*.cxx)
  add_library(${mname} ${TYPE} ${msrcs})
  set_target_properties(${mname} PROPERTIES INSTALL_RPATH "${EXEC_DIR}/lib;${Torch_LINK_DIRECTORIES}")
  target_include_directories(${mname} PUBLIC ${NEML2_SOURCE_DIR}/include ${NEML2_BINARY_DIR}/include)
  target_compile_options(${mname} PRIVATE -Wall -Wextra -pedantic)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${mname} PRIVATE -Werror)
  endif()

  if(NEML2_CPU_PROFILER)
    target_link_libraries(${mname} PUBLIC profiler)
  endif()

  file(GLOB_RECURSE mheaders ${NEML2_SOURCE_DIR}/include/neml2/${mdir}/*.h)
  target_sources(${mname}
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${NEML2_SOURCE_DIR}/include
    FILES
    ${mheaders}
  )
  install(TARGETS ${mname} COMPONENT Development FILE_SET HEADERS)
endmacro()

# ----------------------------------------------------------------------------
# Submodules
# ----------------------------------------------------------------------------
# libneml2_misc
neml2_add_submodule(neml2_misc SHARED misc)
target_include_directories(neml2_misc SYSTEM PUBLIC ${Torch_INCLUDE_DIRECTORIES})
target_link_libraries(neml2_misc PUBLIC ${Torch_LIBRARIES})
target_link_directories(neml2_misc PUBLIC ${Torch_LINK_DIRECTORIES})
target_precompile_headers(neml2_misc PRIVATE ${NEML2_SOURCE_DIR}/include/neml2/misc/types.h)

# libneml2_jit
neml2_add_submodule(neml2_jit SHARED jit)
target_link_libraries(neml2_jit PUBLIC neml2_misc)
target_precompile_headers(neml2_jit PRIVATE ${NEML2_SOURCE_DIR}/include/neml2/jit/types.h)

# libneml2_tensor
neml2_add_submodule(neml2_tensor SHARED tensors)
target_link_libraries(neml2_tensor PUBLIC neml2_jit)
target_precompile_headers(neml2_tensor PRIVATE
  ${NEML2_SOURCE_DIR}/include/neml2/tensors/TensorBase.h
  ${NEML2_SOURCE_DIR}/include/neml2/tensors/TensorBaseImpl.h
  ${NEML2_SOURCE_DIR}/include/neml2/tensors/PrimitiveTensor.h
)

# libneml2_base
neml2_add_submodule(neml2_base SHARED base)
target_link_libraries(neml2_base PUBLIC neml2_tensor)
target_include_directories(neml2_base SYSTEM PUBLIC ${hit_INSTALL_DIR}/include ${wasp_INSTALL_DIR}/include)
target_link_directories(neml2_base PRIVATE ${hit_INSTALL_DIR}/lib ${wasp_INSTALL_DIR}/lib)
target_link_libraries(neml2_base PRIVATE hit waspcore wasphit)

# libneml2_user_tensor
neml2_add_submodule(neml2_user_tensor SHARED user_tensors)
target_link_libraries(neml2_user_tensor PUBLIC neml2_tensor neml2_base)

# libneml2_solver
neml2_add_submodule(neml2_solver SHARED solvers)
target_link_libraries(neml2_solver PUBLIC neml2_tensor neml2_base)

# libneml2_model
neml2_add_submodule(neml2_model SHARED models)
target_link_libraries(neml2_model PUBLIC neml2_tensor neml2_base neml2_solver)
target_precompile_headers(neml2_model PRIVATE ${NEML2_SOURCE_DIR}/include/neml2/models/Model.h)

# libneml2_driver
neml2_add_submodule(neml2_driver SHARED drivers)
target_link_libraries(neml2_driver PUBLIC neml2_tensor neml2_base neml2_model)

# libneml2_runtime
#
# This library is a wrapper of all "runtime" objects to be dynamically loaded.
# Targets who link against neml2_runtime should preferrably use LINK_WHAT_YOU_USE
# to force linking libraries
#
# Otherwise, dlopen-like behavior shall be used to load the runtime objects.
add_library(neml2_runtime INTERFACE)
target_link_libraries(neml2_runtime INTERFACE neml2_base neml2_user_tensor neml2_solver neml2_model neml2_driver)

# ----------------------------------------------------------------------------
# Optional submodules
# ----------------------------------------------------------------------------
# libneml2_dispatcher
if(NEML2_WORK_DISPATCHER)
  neml2_add_submodule(neml2_dispatcher SHARED dispatchers)
  target_link_libraries(neml2_dispatcher PUBLIC neml2_tensor neml2_model)
  target_link_libraries(neml2_dispatcher PUBLIC MPI::MPI_CXX)
  target_link_libraries(neml2_dispatcher PRIVATE timpi_${timpi_BUILD_TYPE})
  target_include_directories(neml2_dispatcher SYSTEM PUBLIC ${timpi_INSTALL_DIR}/include)
  target_link_directories(neml2_dispatcher PRIVATE ${timpi_INSTALL_DIR}/lib)
endif()

# ----------------------------------------------------------------------------
# Export the list of submodules
# ----------------------------------------------------------------------------
set(NEML2_LIBS ${NEML2_LIBS} PARENT_SCOPE)
