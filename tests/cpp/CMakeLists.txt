# ----------------------------------------------------------------------------
# C++ tests for the AOTI runtime dispatcher subsystem
# ----------------------------------------------------------------------------
# Dependency-free: each test is a plain executable that returns non-zero on the
# first failed check (see test_util.h) and is registered as a ctest. No
# third-party test framework is pulled in -- the case count here is small. If it
# grows enough to justify fixtures / rich reporting, revisit then.
#
# Built only under the dev (non-wheel) gate from the top-level CMakeLists, so
# wheel builds never compile or run these.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# --- Pure-logic tests: schedulers + batch slice/cat helpers (no model) -------
foreach(t test_scheduler test_batch_chunk)
      add_executable(${t} ${t}.cpp)
      target_link_libraries(${t} PRIVATE aoti)
      target_compile_options(${t} PRIVATE -Wall -Wextra)
      # Resolve libneml2_aoti + torch at runtime from the build tree.
      set_target_properties(${t} PROPERTIES BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR}")
      add_test(NAME ${t} COMMAND ${t})
      set_tests_properties(${t} PROPERTIES LABELS "dispatcher" TIMEOUT 120)
endforeach()

# --- End-to-end tests: need a compiled artifact ------------------------------
# A ctest fixture compiles the forward_single leaf via `neml2-compile`, emitting
# the standalone-stub layout: <dir>/model_aoti.i + <dir>/model/<device>/...
# - test_dispatcher loads the artifact folder directly (chunk == single shot).
# - test_load_model exercises the C++ load_model(stub, name[, scheduler]) factory.
set(_fixture_dir ${CMAKE_CURRENT_BINARY_DIR}/dispatch_fixture)
set(_fixture_input ${NEML2_SOURCE_DIR}/tests/aoti/forward_single/model.i)
set(_artifact_dir ${_fixture_dir}/model)        # <dir>/<model>/<device>/...
set(_stub ${_fixture_dir}/model_aoti.i)         # standalone stub next to it

# Compile cpu always; add cuda when libtorch has CUDA so the tests can exercise
# the cross-device path. The tests runtime-gate on at::hasCUDA() + the cuda
# artifact existing, so this is a no-op on cpu-only machines.
set(_fixture_devices cpu)
if(TARGET torch::cuda)
      list(APPEND _fixture_devices cuda)
endif()

add_test(
      NAME dispatcher_fixture_compile
      COMMAND ${Python3_EXECUTABLE} -m neml2.cli.aoti_compile ${_fixture_input}
              --model model --device ${_fixture_devices} --output-dir ${_fixture_dir}
      WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
)
set_tests_properties(dispatcher_fixture_compile PROPERTIES
      FIXTURES_SETUP dispatch_artifact
      LABELS "dispatcher"
      TIMEOUT 600
      ENVIRONMENT "TORCHINDUCTOR_CACHE_DIR=${_fixture_dir}/aoti-cache"
)

# Loading a .pt2 dlopen's its internal compiled .so, which links libtorch.so.
# That transitive dlopen resolves NEEDED libs via the loader's search path, not
# the exe's DT_RUNPATH, so torch/lib must be on LD_LIBRARY_PATH at run time.
foreach(t test_dispatcher test_load_model)
      add_executable(${t} ${t}.cpp)
      target_link_libraries(${t} PRIVATE aoti)
      target_compile_options(${t} PRIVATE -Wall -Wextra)
      set_target_properties(${t} PROPERTIES BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR}")
endforeach()

# test_dispatcher takes the artifact folder; test_load_model takes the fixture
# (collection) dir and derives the stub + per-device meta from it.
add_test(NAME test_dispatcher COMMAND test_dispatcher ${_artifact_dir})
add_test(NAME test_load_model COMMAND test_load_model ${_fixture_dir})
set_tests_properties(test_dispatcher test_load_model PROPERTIES
      FIXTURES_REQUIRED dispatch_artifact
      LABELS "dispatcher"
      TIMEOUT 300
      ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${torch_LINK_DIR}"
)
