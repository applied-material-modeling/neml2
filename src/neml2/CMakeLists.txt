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
include(SetupWASP)

# ----------------------------------------------------------------------------
# HIT
# ----------------------------------------------------------------------------
include(SetupHIT)

# ----------------------------------------------------------------------------
# Work dispatcher (MPI and TIMPI)
# ----------------------------------------------------------------------------
if(NEML2_WORK_DISPATCHER)
  find_package(MPI REQUIRED)

  if(NOT MPI_C_FOUND)
    message(FATAL_ERROR "MPI C compiler (required for work dispatcher) not found")
  endif()

  if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "MPI C++ compiler (required for work dispatcher) not found")
  endif()

  include(SetupTIMPI)
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
target_link_libraries(neml2_base PUBLIC neml2_tensor HIT WASP)

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
  target_link_libraries(neml2_dispatcher PUBLIC MPI::MPI_CXX TIMPI)
endif()

# ----------------------------------------------------------------------------
# Export the list of submodules
# ----------------------------------------------------------------------------
set(NEML2_LIBS ${NEML2_LIBS} PARENT_SCOPE)
