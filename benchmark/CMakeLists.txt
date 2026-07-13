# ----------------------------------------------------------------------------
# Benchmark ctest registration
# ----------------------------------------------------------------------------
# One ctest entry per benchmark/<scenario>/model.i. Each ctest invokes
# `python -m benchmark.run_benchmark <scenario> --batch N --run-batch M`,
# which compiles the [Drivers]/driver block through neml2-compile at
# batch=N (the AOTI example shape), then loads the stub and runs the
# driver at batch=M. When N != M this exercises the .pt2's dynamic-batch
# generalisability: ``torch.export`` marks the leading batch dim as a
# dynamic Dim, so the compiled graph must accept any positive batch at
# runtime.
#
# Default policy: compile=2, run=8 -- the smallest pair that probes
# dynamic-Dim generality without spending the autotune budget on a
# large workload.
#
# EXCEPTION: scenarios where ${nbatch} ALSO drives a STATIC sub-batch
# axis (currently just ``mxpc``) must compile and run at the same value
# because the sub-batch dim is baked into the AOTI graph at trace time.
# These scenarios get a fixed compile==run via BENCH_STATIC_SUBBATCH.
#
# run_benchmark.py reads the driver block's `model = '...'` field to figure
# out what to compile -- no per-scenario --model override map is needed,
# even for scenarios where the driver points at an outer ComposedModel
# wrapper (e.g. crystal-plasticity's `model_with_stress`).
#
# Compiled .pt2 artifacts land in ${CMAKE_BINARY_DIR}/benchmark/<scenario>/.
# Each scenario gets its OWN build-tree-local TORCHINDUCTOR_CACHE_DIR so the
# tests can run concurrently under ``ctest --parallel`` without racing on a
# shared cache; the per-scenario cache still persists across re-runs, so a
# repeated single-scenario run stays warm. Heavy scenarios declare
# ``PROCESSORS 2`` so ``ctest -j`` doesn't pack too many AOTI compiles (each
# spawns a multi-threaded C++ build) onto the runner at once.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GLOB BENCH_INPUTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/*/model.i)

# Heavy scenarios: AOTI compile + driver run can take several minutes at
# small batch because the compile dominates wall time. Even "cheap" scenarios
# run multiple minutes under `ctest --parallel 4` (4 concurrent Inductor
# compiles contend for the runner), so the timeouts below carry generous
# headroom rather than tracking the no-contention wall time.
set(BENCH_HEAVY chaboche6 chaboche12 chaboche12gmres scpcoup scpcoupmult scpdecoup scpdecoupexp tcprandom tcpsingle)

# Scenarios where ${nbatch} drives a static sub-batch axis (not the
# dynamic batch dim). Sub-batch shapes are baked at trace time so the
# runtime tensor must match -- runtime != compile would be a shape
# mismatch, not a dynamic-Dim generality check. Skip the run-batch
# differential for these.
set(BENCH_STATIC_SUBBATCH mxpc)

foreach(bench_input ${BENCH_INPUTS})
      get_filename_component(scenario ${bench_input} DIRECTORY)
      set(work_dir ${CMAKE_BINARY_DIR}/benchmark/${scenario})
      file(MAKE_DIRECTORY ${work_dir})

      if(${scenario} IN_LIST BENCH_STATIC_SUBBATCH)
            set(_bench_args --batch 2)
      else()
            set(_bench_args --batch 2 --run-batch 8)
      endif()

      add_test(
            NAME benchmark_${scenario}
            COMMAND ${Python3_EXECUTABLE} -m benchmark.run_benchmark ${scenario}
                    ${_bench_args} --device cpu --driver driver
                    --output-dir ${work_dir}
            WORKING_DIRECTORY ${NEML2_SOURCE_DIR}
      )

      if(${scenario} IN_LIST BENCH_HEAVY)
            set(_t 2400)
            set(_procs 2)
      else()
            # 1200s, not 600s: under --parallel 4 a "cheap" scenario's Inductor
            # compile contends with three others and has been observed to exceed
            # 600s on CI (e.g. chaboche4).
            set(_t 1200)
            set(_procs 1)
      endif()

      set_tests_properties(benchmark_${scenario} PROPERTIES
            LABELS "benchmark"
            TIMEOUT ${_t}
            PROCESSORS ${_procs}
            SKIP_RETURN_CODE 77
            ENVIRONMENT "TORCHINDUCTOR_CACHE_DIR=${work_dir}/aoti-cache"
      )
endforeach()
