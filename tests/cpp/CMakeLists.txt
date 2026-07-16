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

# /wd4251;/wd4275 mirror the aoti target (top-level CMakeLists): these exes
# include the exported PImpl facade headers, so under /W3 MSVC would otherwise
# emit the same "needs dll-interface" noise for the std:: members carried across
# the boundary (harmless -- same toolchain/runtime on both sides).
function(neml2_add_test_warning_flags target)
      target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>
            $<$<CXX_COMPILER_ID:MSVC>:/W3;/wd4251;/wd4275>)
endfunction()

# torch Inductor nests two ~52-char hash directories (plus a ~52-char filename)
# under TORCHINDUCTOR_CACHE_DIR. Rooted at <fixture_dir>/aoti-cache deep in the
# build tree, the resulting paths exceed Windows' 260-char MAX_PATH and MASM
# (ml64.exe) fails with "fatal error A1000: cannot open file". On Windows root
# the per-fixture caches in the short system temp dir instead; elsewhere keep
# them build-tree-local so `ctest --parallel` runs stay isolated per fixture.
function(neml2_inductor_cache_dir out_var fixture_name default_dir)
      if(WIN32)
            # cmake_path(SET), not file(TO_CMAKE_PATH): the latter treats its
            # input as a search-path list and rewrites the "C:" drive colon to a
            # ";", which then splits the ENVIRONMENT test property in two -- the
            # cache would silently fall back to a relative "C" dir under the deep
            # build tree, defeating this whole shortening.
            cmake_path(SET _tmp "$ENV{TEMP}")
            set(${out_var} "${_tmp}/neml2-ic/${fixture_name}" PARENT_SCOPE)
      else()
            set(${out_var} "${default_dir}/aoti-cache" PARENT_SCOPE)
      endif()
endfunction()

# --- Pure-logic tests: schedulers + batch slice/cat helpers + exceptions + log +
# the masked-Newton substep_del_tol convergence gate (hand-built NonlinearSystem,
# no compiled artifact) -----------------------------------------------------------
foreach(t test_scheduler test_batch_chunk test_static_hybrid_scheduler test_exceptions test_log
          test_newton_substep_del_tol)
      add_executable(${t} ${t}.cpp)
      target_link_libraries(${t} PRIVATE aoti)
      neml2_add_test_warning_flags(${t})
      # Resolve libneml2 + torch at runtime from the build tree.
      set_target_properties(${t} PROPERTIES BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR}")
      add_test(NAME ${t} COMMAND ${t})
      set_tests_properties(${t} PROPERTIES LABELS "dispatcher" TIMEOUT 120)
endforeach()

# --- Krylov linear solvers: batched GMRES / BiCGStab numerics (krylov.h) ------
# Pure numerics: an in-memory dense matvec (no compiled artifact, no
# NonlinearSystem) validated against at::linalg_solve. `krylov.h` is header-only
# so nothing new links; its own `krylov` label runs it via `ctest -L krylov`.
add_executable(test_krylov test_krylov.cpp)
target_link_libraries(test_krylov PRIVATE aoti)
neml2_add_test_warning_flags(test_krylov)
set_target_properties(test_krylov PROPERTIES BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR}")
add_test(NAME test_krylov COMMAND test_krylov)
set_tests_properties(test_krylov PROPERTIES LABELS "krylov" TIMEOUT 120)

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
              --model model --device ${_fixture_devices} -p model.E -p model.nu
              -d : -d stress:model.E -d stress:model.nu
              --output-dir ${_fixture_dir}
      WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
)
neml2_inductor_cache_dir(_fixture_cache dispatch_fixture ${_fixture_dir})
set_tests_properties(dispatcher_fixture_compile PROPERTIES
      FIXTURES_SETUP dispatch_artifact
      LABELS "dispatcher"
      TIMEOUT 600
      ENVIRONMENT "TORCHINDUCTOR_CACHE_DIR=${_fixture_cache}"
)

# Loading a .pt2 dlopen's its internal compiled .so, which links libtorch.so.
# That transitive dlopen resolves NEEDED libs via the loader's search path, not
# the exe's DT_RUNPATH, so torch/lib must be on LD_LIBRARY_PATH at run time.
foreach(t test_dispatcher test_load_model)
      add_executable(${t} ${t}.cpp)
      target_link_libraries(${t} PRIVATE aoti)
      neml2_add_test_warning_flags(${t})
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

# --- Boundary renames: a renamed artifact reports the boundary names on the C++
# Model facade + DispatchedModel, and dispatch parity is preserved. ------------
# Same forward_promoted leaf as the dispatcher fixture, but compiled with the
# three --rename-* flags so the authored strain/stress/model.{E,nu} surface as
# eps/sig/youngs/poisson. Reuses _fixture_devices (cpu [+cuda]).
set(_ren_dir ${CMAKE_CURRENT_BINARY_DIR}/rename_fixture)
set(_ren_artifact_dir ${_ren_dir}/model)

add_test(
      NAME rename_fixture_compile
      COMMAND ${Python3_EXECUTABLE} -m neml2.cli.aoti_compile ${_fixture_input}
              --model model --device ${_fixture_devices} -p model.E -p model.nu
              -d : -d stress:model.E -d stress:model.nu
              --rename-input strain:eps --rename-output stress:sig
              --rename-parameter model.E:youngs --rename-parameter model.nu:poisson
              --output-dir ${_ren_dir}
      WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
)
neml2_inductor_cache_dir(_ren_cache rename_fixture ${_ren_dir})
set_tests_properties(rename_fixture_compile PROPERTIES
      FIXTURES_SETUP rename_artifact
      LABELS "dispatcher"
      TIMEOUT 600
      ENVIRONMENT "TORCHINDUCTOR_CACHE_DIR=${_ren_cache}"
)

add_executable(test_dispatcher_rename test_dispatcher_rename.cpp)
target_link_libraries(test_dispatcher_rename PRIVATE aoti)
neml2_add_test_warning_flags(test_dispatcher_rename)
set_target_properties(test_dispatcher_rename PROPERTIES
      BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR}")
add_test(NAME test_dispatcher_rename COMMAND test_dispatcher_rename ${_ren_artifact_dir})
set_tests_properties(test_dispatcher_rename PROPERTIES
      FIXTURES_REQUIRED rename_artifact
      LABELS "dispatcher"
      TIMEOUT 300
      ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${torch_LINK_DIR}"
)

# --- Custom-op coverage: a model that uses neml2::opaque_pow -------------------
# The dispatcher fixtures above compile LinearIsotropicElasticity (no custom ops),
# so nothing exercised the cpp-aoti path for an op that torch.export preserves
# into the artifact and that the Python-free libneml2 must register itself
# (neml2::opaque_pow -- crystal plasticity / Perzyna). PerzynaPlasticFlowRate is
# the smallest plain-batch leaf that uses it. Reuses _fixture_devices (cpu [+cuda]).
set(_pz_dir ${CMAKE_CURRENT_BINARY_DIR}/custom_op_fixture)
set(_pz_input ${NEML2_SOURCE_DIR}/tests/cpp/fixtures/opaque_pow_model.i)

add_test(
      NAME custom_op_fixture_compile
      COMMAND ${Python3_EXECUTABLE} -m neml2.cli.aoti_compile ${_pz_input}
              --model model --device ${_fixture_devices} --dtype float64
              -d flow_rate:yield_function
              --output-dir ${_pz_dir}
      WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
)
neml2_inductor_cache_dir(_pz_cache custom_op_fixture ${_pz_dir})
set_tests_properties(custom_op_fixture_compile PROPERTIES
      FIXTURES_SETUP custom_op_artifact
      LABELS "dispatcher"
      TIMEOUT 600
      ENVIRONMENT "TORCHINDUCTOR_CACHE_DIR=${_pz_cache}"
)

add_executable(test_custom_ops test_custom_ops.cpp)
target_link_libraries(test_custom_ops PRIVATE aoti)
neml2_add_test_warning_flags(test_custom_ops)
set_target_properties(test_custom_ops PROPERTIES BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR}")
add_test(NAME test_custom_ops COMMAND test_custom_ops ${_pz_dir})
set_tests_properties(test_custom_ops PROPERTIES
      FIXTURES_REQUIRED custom_op_artifact
      LABELS "dispatcher"
      TIMEOUT 300
      ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${torch_LINK_DIR}"
)

# --- Input broadcast: a batch-independent input alongside a batched one --------
# Every AOTI fixture above feeds uniformly-batched inputs, so the runtime's
# broadcast of a batch-independent input (the shape MOOSE produces for a scalar
# TIME force) was uncovered. ScalarLinearCombination C = A + B + offset, with B
# supplied as a 0-dim scalar. Reuses _fixture_devices (cpu [+cuda]).
set(_bc_dir ${CMAKE_CURRENT_BINARY_DIR}/broadcast_fixture)
set(_bc_input ${NEML2_SOURCE_DIR}/tests/cpp/fixtures/batch_broadcast_model.i)

add_test(
      NAME broadcast_fixture_compile
      COMMAND ${Python3_EXECUTABLE} -m neml2.cli.aoti_compile ${_bc_input}
              --model model --device ${_fixture_devices} --dtype float64
              -d C:A -d C:B
              --output-dir ${_bc_dir}
      WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
)
neml2_inductor_cache_dir(_bc_cache broadcast_fixture ${_bc_dir})
set_tests_properties(broadcast_fixture_compile PROPERTIES
      FIXTURES_SETUP broadcast_artifact
      LABELS "dispatcher"
      TIMEOUT 600
      ENVIRONMENT "TORCHINDUCTOR_CACHE_DIR=${_bc_cache}"
)

add_executable(test_input_broadcast test_input_broadcast.cpp)
target_link_libraries(test_input_broadcast PRIVATE aoti)
neml2_add_test_warning_flags(test_input_broadcast)
set_target_properties(test_input_broadcast PROPERTIES BUILD_RPATH "${NEML2_BINARY_DIR};${torch_LINK_DIR}")
add_test(NAME test_input_broadcast COMMAND test_input_broadcast ${_bc_dir})
set_tests_properties(test_input_broadcast PROPERTIES
      FIXTURES_REQUIRED broadcast_artifact
      LABELS "dispatcher"
      TIMEOUT 300
      ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${torch_LINK_DIR}"
)

# --- Eager embed test: links libneml2_eager, embeds a CPython interpreter, and
# runs a model straight from the original .i (no compile fixture needed). New
# "eager" label so it runs independently of the AOTI dispatcher tests.
# Skipped on Windows: libneml2_eager is not built there (the undefined-symbol
# embed linking model is Unix-only -- see the top-level CMakeLists).
if(NOT WIN32)
add_executable(test_eager test_eager.cpp)
# This pure-C++ exe supplies libpython (Python3::Python) for the interpreter that
# libneml2_eager embeds; libneml2_eager itself links only Python3::Module.
target_link_libraries(test_eager PRIVATE eager aoti Python3::Python)
neml2_add_test_warning_flags(test_eager)
# libneml2_eager deliberately leaves the torch_python (+ CPython) symbols undefined
# for the host process to supply at load time, so the wheel never has to ship
# libpython. Linking that .so into an executable therefore needs the linker to
# tolerate those undefined refs (resolved at runtime via BUILD_RPATH to torch/lib +
# the linked libpython). Most system linkers allow this by default, but some -- e.g.
# conda's compiler-compat ld used by the pip build -- default to
# --no-allow-shlib-undefined and must be told explicitly.
if(APPLE)
      target_link_options(test_eager PRIVATE LINKER:-undefined,dynamic_lookup)
else()
      target_link_options(test_eager PRIVATE LINKER:--allow-shlib-undefined)
endif()
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
# argv[3]/argv[4]: an .i using an out-of-package model type + the external module
# defining it, exercising the `--load` extension hook.
add_test(NAME test_eager
      COMMAND test_eager
            ${NEML2_SOURCE_DIR}/tests/aoti/forward_single/model.i
            ${NEML2_SOURCE_DIR}/tests/cpp/fixtures/implicit_diverge.i
            ${NEML2_SOURCE_DIR}/tests/cpp/fixtures/ext_model.i
            ${NEML2_SOURCE_DIR}/tests/cpp/fixtures/ext_model.py)
# The embedded interpreter must find the stdlib (PYTHONHOME) + the neml2 source
# package (PYTHONPATH); torch/lib stays on LD_LIBRARY_PATH for the transitive
# libtorch resolution the existing tests also rely on.
set_tests_properties(test_eager PROPERTIES
      LABELS "eager"
      TIMEOUT 300
      ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${torch_LINK_DIR}"
      ENVIRONMENT "PYTHONHOME=${_py_home};PYTHONPATH=${NEML2_SOURCE_DIR}"
)
endif() # NOT WIN32 -- test_eager (libneml2_eager unavailable on Windows)
