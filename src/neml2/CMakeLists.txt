# ----------------------------------------------------------------------------
# Torch
# ----------------------------------------------------------------------------
find_package(Torch)

if(NOT Torch_FOUND)
  include(DownloadTorch)
  set(Torch_ROOT ${torch_SOURCE_DIR} CACHE PATH "Torch root directory" FORCE)
  find_package(Torch REQUIRED)
endif()

if(HAS_GLIBC)
  if(${GLIBCXX_USE_CXX11_ABI} STREQUAL "0")
    message(STATUS "Using pre C++11 ABI")
  else()
    message(STATUS "Using C++11 ABI")
  endif()
endif()

cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${Torch_LINK_DIR} Torch_DOWNLOADED)

if(Torch_DOWNLOADED)
  install(DIRECTORY ${Torch_INCLUDE_DIR} TYPE INCLUDE COMPONENT libneml2)
  install(DIRECTORY ${Torch_LINK_DIR} DESTINATION . COMPONENT libneml2)
endif()

# ----------------------------------------------------------------------------
# HIT
# ----------------------------------------------------------------------------
find_package(HIT)

if(NOT HIT_FOUND)
  # ----------------------------------------------------------------------------
  # WASP
  # ----------------------------------------------------------------------------
  if(NOT WASP_FOUND)
    include(DownloadWASP)
    set(WASP_ROOT ${WASP_INSTALL_DIR} CACHE PATH "WASP root directory" FORCE)
    find_package(WASP REQUIRED)
  endif()

  include(DownloadHIT)
  set(HIT_ROOT ${HIT_INSTALL_DIR} CACHE PATH "HIT root directory" FORCE)
  find_package(HIT REQUIRED)
endif()

cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${WASP_LINK_DIR} WASP_DOWNLOADED)
cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${HIT_LINK_DIR} HIT_DOWNLOADED)

if(WASP_DOWNLOADED)
  install(DIRECTORY ${WASP_INCLUDE_DIR}/waspcore ${WASP_INCLUDE_DIR}/wasphit TYPE INCLUDE COMPONENT libneml2)
endif()

if(HIT_DOWNLOADED)
  install(DIRECTORY ${HIT_INCLUDE_DIR}/hit TYPE INCLUDE COMPONENT libneml2)
endif()

# ----------------------------------------------------------------------------
# TIMPI
# ----------------------------------------------------------------------------
if(NEML2_WORK_DISPATCHER)
  find_package(TIMPI)

  if(NOT TIMPI_FOUND)
    include(DownloadTIMPI)
    set(TIMPI_ROOT ${TIMPI_INSTALL_DIR} CACHE PATH "TIMPI root directory" FORCE)
    find_package(TIMPI REQUIRED)
  endif()

  cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${TIMPI_LINK_DIR} TIMPI_DOWNLOADED)

  if(TIMPI_DOWNLOADED)
    install(DIRECTORY ${TIMPI_INCLUDE_DIR}/timpi TYPE INCLUDE COMPONENT libneml2)
  endif()
endif()

# ----------------------------------------------------------------------------
# CPU Profiler
# ----------------------------------------------------------------------------
if(NEML2_CPU_PROFILER)
  find_package(Gperftools)

  if(NOT Gperftools_FOUND)
    include(DownloadGperftools)
    set(Gperftools_ROOT ${GPERFTOOLS_INSTALL_DIR} CACHE PATH "gperftools root directory" FORCE)
    find_package(Gperftools REQUIRED)
  endif()

  cmake_path(IS_PREFIX CMAKE_BINARY_DIR ${Gperftools_LINK_DIR} Gperftools_DOWNLOADED)

  if(Gperftools_DOWNLOADED)
    # libneml2 does not need gperftools headers
  endif()
endif()

# ----------------------------------------------------------------------------
# nlohmann json
# ----------------------------------------------------------------------------
set(JSON_MultipleHeaders OFF)
find_package(nlohmann_json QUIET)

if(NOT TARGET nlohmann_json::nlohmann_json)
  include(FetchContent)
  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v${NLOHMANN_JSON_VERSION}/json.tar.xz)
  message(STATUS "Downloading/updating nlohmann/json")
  set(JSON_Install ON)
  set(JSON_SystemInclude ON)
  FetchContent_MakeAvailable(json)
endif()

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
    BASE_DIRS ${NEML2_SOURCE_DIR}/include
    FILES
    ${mheaders}
  )

  # Append suffix to the output name (only if the build type is not Release)
  set_target_properties(${mname} PROPERTIES OUTPUT_NAME "${mname}$<IF:$<CONFIG:Release>,,_$<CONFIG>>")

  target_compile_options(${mname} PRIVATE -Wall -Wextra -pedantic)

  if(HAS_GLIBC)
    target_compile_options(${mname} PUBLIC -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})
  endif()

  if(NEML2_CPU_PROFILER)
    target_link_libraries(${mname} PRIVATE Gperftools::profiler)
  endif()

  if(NEML2_THREAD_SANITIZER)
    target_compile_options(${mname} PUBLIC ${CMAKE_CXX_FLAGS_THREAD_SANITIZER})
    target_link_options(${mname} PUBLIC ${CMAKE_CXX_FLAGS_THREAD_SANITIZER})
  endif()

  install(TARGETS ${mname} FILE_SET HEADERS COMPONENT libneml2)
  install(TARGETS ${mname} LIBRARY COMPONENT libneml2)
endfunction()

# ----------------------------------------------------------------------------
# Submodules
# ----------------------------------------------------------------------------
# libneml2
#
# This library is a wrapper of all neml2 library components.
#
# Since neml2 relies on dynamic loading of object symbols, targets who link
# against neml2 should preferrably use LINK_WHAT_YOU_USE. Otherwise, dlopen-like
# utilities shall be used to load the runtime objects.
add_library(neml2 INTERFACE)

# libneml2_misc
neml2_add_submodule(neml2_misc SHARED misc)
target_link_libraries(neml2_misc PUBLIC Torch::Torch nlohmann_json::nlohmann_json)
set_target_properties(neml2_misc PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
target_link_libraries(neml2 INTERFACE neml2_misc)

if(NEML2_PCH)
  target_precompile_headers(neml2_misc PUBLIC
    ${NEML2_SOURCE_DIR}/include/neml2/misc/types.h
    ${NEML2_SOURCE_DIR}/include/neml2/misc/assertions.h
  )
endif()

# libneml2_jit
neml2_add_submodule(neml2_jit SHARED jit)
target_link_libraries(neml2_jit PUBLIC neml2_misc)
target_link_libraries(neml2 INTERFACE neml2_jit)

if(NEML2_PCH)
  target_precompile_headers(neml2_jit PUBLIC ${NEML2_SOURCE_DIR}/include/neml2/jit/types.h)
endif()

# libneml2_tensor
neml2_add_submodule(neml2_tensor SHARED tensors)
target_link_libraries(neml2_tensor PUBLIC neml2_jit)
target_link_libraries(neml2 INTERFACE neml2_tensor)

if(NEML2_PCH)
  target_precompile_headers(neml2_tensor PUBLIC
    ${NEML2_SOURCE_DIR}/include/neml2/tensors/TensorBase.h
    ${NEML2_SOURCE_DIR}/include/neml2/tensors/TensorBaseImpl.h
    ${NEML2_SOURCE_DIR}/include/neml2/tensors/PrimitiveTensor.h
    ${NEML2_SOURCE_DIR}/include/neml2/tensors/assertions.h
    ${NEML2_SOURCE_DIR}/include/neml2/tensors/shape_utils.h
  )
endif()

# libneml2_base
neml2_add_submodule(neml2_base SHARED base)
target_link_libraries(neml2_base PUBLIC neml2_tensor HIT)
set_target_properties(neml2_base PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
target_link_libraries(neml2 INTERFACE neml2_base)

if(NEML2_PCH)
  target_precompile_headers(neml2_base PUBLIC
    ${NEML2_SOURCE_DIR}/include/neml2/base/Option.h
    ${NEML2_SOURCE_DIR}/include/neml2/base/TensorName.h
  )
endif()

# libneml2_user_tensor
neml2_add_submodule(neml2_user_tensor SHARED user_tensors)
target_link_libraries(neml2_user_tensor PUBLIC neml2_tensor neml2_base)
target_link_options(neml2_user_tensor INTERFACE ${CMAKE_CXX_LINK_WHAT_YOU_USE_FLAG})
target_link_libraries(neml2 INTERFACE neml2_user_tensor)

# libneml2_solver
neml2_add_submodule(neml2_solver SHARED solvers)
target_link_libraries(neml2_solver PUBLIC neml2_tensor neml2_base)
target_link_options(neml2_solver INTERFACE ${CMAKE_CXX_LINK_WHAT_YOU_USE_FLAG})
target_link_libraries(neml2 INTERFACE neml2_solver)

# libneml2_model
neml2_add_submodule(neml2_model SHARED models)
target_link_libraries(neml2_model PUBLIC neml2_solver neml2_tensor neml2_base)
target_link_options(neml2_model INTERFACE ${CMAKE_CXX_LINK_WHAT_YOU_USE_FLAG})
target_link_libraries(neml2 INTERFACE neml2_model)

if(NEML2_PCH)
  target_precompile_headers(neml2_model PUBLIC
    ${NEML2_SOURCE_DIR}/include/neml2/models/Model.h
  )
endif()

# libneml2_driver
neml2_add_submodule(neml2_driver SHARED drivers)
target_link_libraries(neml2_driver PUBLIC neml2_model neml2_tensor neml2_base)
target_link_options(neml2_driver INTERFACE ${CMAKE_CXX_LINK_WHAT_YOU_USE_FLAG})
target_link_libraries(neml2 INTERFACE neml2_driver)

# libneml2_dispatcher
if(NEML2_WORK_DISPATCHER)
  neml2_add_submodule(neml2_dispatcher SHARED dispatchers)
  target_link_libraries(neml2_dispatcher PUBLIC neml2_tensor neml2_model)
  target_link_libraries(neml2_dispatcher PUBLIC TIMPI)
  target_compile_definitions(neml2_dispatcher PUBLIC NEML2_HAS_DISPATCHER)
  set_target_properties(neml2_dispatcher PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
  target_link_libraries(neml2_driver PUBLIC neml2_dispatcher)
  target_link_libraries(neml2 INTERFACE neml2_dispatcher)
endif()

# ###############################################################################
# Version number and hash
# ###############################################################################
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
