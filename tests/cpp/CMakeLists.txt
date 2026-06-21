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

# Interpreter: drives the AOTI compile fixtures (neml2-compile). Development.Embed
# (libpython): linked into the pure-C++ test_eager exe so it can host the embedded
# interpreter -- libneml2_eager itself only links Python3::Module (see the top-level
# CMakeLists), leaving the CPython symbols to be supplied by whatever embeds it.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed)

# --- Pure-logic tests: schedulers + batch slice/cat helpers + exceptions -----
foreach(t test_scheduler test_batch_chunk test_static_hybrid_scheduler test_exceptions)
      add_executable(${t} ${t}.cpp)
      target_link_libraries(${t} PRIVATE aoti)
      target_compile_options(${t} PRIVATE -Wall -Wextra)
      # Resolve libneml2 + torch at runtime from the build tree.
      set_target_properties(${t} PROPERTIES BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR}")
      add_test(NAME ${t} COMMAND ${t})
      set_tests_properties(${t} PROPERTIES LABELS "dispatcher" TIMEOUT 120)
endforeach()

# --- End-to-end tests: need a compiled artifact ------------------------------
# A ctest fixture compiles the forward_promoted leaf (LinearIsotropicElasticity
# with E + nu promoted to mutable parameters) via `neml2-compile`, emitting the
# standalone-stub layout: <dir>/model_aoti.i + <dir>/model/<device>/...
# - test_dispatcher loads the artifact folder directly (chunk == single shot,
#   plus the async + GPU-hybrid + write-through paths).
# - test_load_model exercises the C++ load_model(stub, name[, scheduler]) factory.
# Promoting E/nu doesn't change the default-parameter forward output, so the
# single-shot comparisons are unaffected; it just lets the write-through test
# mutate a parameter.
set(_fixture_dir ${CMAKE_CURRENT_BINARY_DIR}/dispatch_fixture)
set(_fixture_input ${NEML2_SOURCE_DIR}/tests/aoti/forward_promoted/model.i)
set(_artifact_dir ${_fixture_dir}/model)        # <dir>/<model>/<device>/...
set(_stub ${_fixture_dir}/model_aoti.i)         # standalone stub next to it

# Compile cpu always; add cuda when libtorch has CUDA so the tests can exercise
# the cross-device + hybrid paths. The tests runtime-gate on at::hasCUDA() + the
# cuda artifact existing, so this is a no-op on cpu-only machines.
set(_fixture_devices cpu)
if(TARGET torch::cuda)
      list(APPEND _fixture_devices cuda)
endif()

add_test(
      NAME dispatcher_fixture_compile
      COMMAND ${Python3_EXECUTABLE} -m neml2.cli.aoti_compile ${_fixture_input}
              --model model --device ${_fixture_devices} -p E -p nu -d :
              --output-dir ${_fixture_dir}
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

# --- Eager embed test: links libneml2_eager, embeds a CPython interpreter, and
# runs a model straight from the original .i (no compile fixture needed). New
# "eager" label so it runs independently of the AOTI dispatcher tests.
add_executable(test_eager test_eager.cpp)
# This pure-C++ exe supplies libpython (Python3::Python) for the interpreter that
# libneml2_eager embeds; libneml2_eager itself links only Python3::Module.
target_link_libraries(test_eager PRIVATE eager aoti Python3::Python)
target_compile_options(test_eager PRIVATE -Wall -Wextra)
# RUNPATH is not transitive, so the test exe's own rpath must reach libneml2 +
# torch + libpython.
get_filename_component(_py_bindir "${Python3_EXECUTABLE}" DIRECTORY)
get_filename_component(_py_home "${_py_bindir}" DIRECTORY)
set(_py_libdir "${_py_home}/lib")
set_target_properties(test_eager PROPERTIES
      BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR};${_py_libdir}")
# argv[1]: the forward fixture (forward / jvp / jacobian / device-override).
# argv[2]: a max_its=0 implicit fixture driving the recoverable ConvergenceError
# path (kept under tests/cpp so the AOTI compile sweep never drives it).
add_test(NAME test_eager
      COMMAND test_eager
            ${NEML2_SOURCE_DIR}/tests/aoti/forward_single/model.i
            ${NEML2_SOURCE_DIR}/tests/cpp/fixtures/implicit_diverge.i)
# The embedded interpreter must find the stdlib (PYTHONHOME) + the neml2 source
# package (PYTHONPATH); torch/lib stays on LD_LIBRARY_PATH for the transitive
# libtorch resolution the existing tests also rely on.
set_tests_properties(test_eager PROPERTIES
      LABELS "eager"
      TIMEOUT 300
      ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${torch_LINK_DIR}"
      ENVIRONMENT "PYTHONHOME=${_py_home};PYTHONPATH=${NEML2_SOURCE_DIR}"
)
