# Python-native NEML2

Python-native material model surface used by the `torch.export` + AOTInductor
migration. See `MIGRATION.md` at the repo root for the phase plan and
`DECISION.md` for the design decisions. This README is
strictly a kickstart — layout, environment, build, tests, and benchmark
results.

## Layout

- `types/` — typed tensor wrappers (`Scalar`, `SR2`, `SSR4`) over raw
  `torch.Tensor` storage, registered as pytree nodes for `torch.export`.
  `types/functions.py` carries free ops (`tr`, `vol`, `dev`, `norm`, …).
- `chain_rule.py` — type aliases for chain-rule propagation (`ChainRuleDict`,
  `ChainRuleAction`, `ListDerivSpec`, `SparsityFlag`).
- `model.py` — `Model(nn.Module)`,
  `register_typed_buffer` / `register_typed_parameter`, `apply_chain_rule`.
- `resolver.py` — `DependencyResolver`.
- `equation_systems.py` — `AxisLayout`, `IStructure`, `AssembledVector/Matrix`,
  `ModelNonlinearSystem`, plus dense export wrappers (`DenseRHS`,
  `DenseOperator`, `DenseLinearizedSystem`, `DenseImplicitSensitivity`,
  `DenseNewtonStep`, `DenseIFT`).
- `solvers.py` — `DenseLU`, `SchurComplement` (block 2-group factorisation), `Newton`.
- `factory.py` — HIT input parsing (`load_input`, `load_model`,
  `register_native`, `_NativeInputFile.get_tensor`).
- `data/` — registered `[Data]` objects: `CrystalGeometry` (base) and
  `CubicCrystal`.
- `models/` — registered leaves, one file per C++ header; the composition
  infrastructure `models/common/ComposedModel.py` (`ComposedModel`;
  dependency-resolved composition) and `models/common/ImplicitUpdate.py`
  (`ImplicitUpdate(Model)` with `torch.autograd.Function`-backed IFT-style
  backward) live alongside them.
- `export.py` — `compile_model(model, example_inputs, package_path)`.
- `aoti_export.py` — `export_model_for_aoti(...)`; metadata sidecar +
  segment `.pt2`s. Schema documented in
  [`AOTI_PACKAGES.md`](AOTI_PACKAGES.md).

## Environment

- `torch == 2.12.0` (per `dependencies.yaml`).
- `nmhit >= 0.2.0`.
- `torch._inductor.aoti_compile_and_package` + `aoti_load_package` importable.
- C++ build uses the AOTI package-loader header from the libtorch found by
  `cmake/Modules/Findtorch.cmake`.

**CUDA AOTI export** needs a full toolkit on `CUDA_HOME` (Inductor's
cpp_wrapper compiles a host launcher that `#include`s `crt/host_defines.h`;
the pip `nvidia-cuda-runtime-cuXX` packages don't have the `crt/`
subdirectory):

```bash
export CUDA_HOME=/usr/local/cuda-12.X
# or via conda-installed nvcc:
# conda install -c nvidia cuda-nvcc=<matching-version>
# export CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux
```

CPU AOTI export needs no `CUDA_HOME`.

## Build And Run

```bash
# Editable install picks up the Python-native package.
pip install -e ".[dev]" -v

# C++ AOTI harness (Catch2 binary).
cmake --preset aoti -GNinja
cmake --build --preset aoti -j$(nproc)
```

The interpreter for the lazy AOTI export subprocess defaults to `python` on
`PATH`. Override with `NEML2_PYTHON_EXECUTABLE=/path/to/python` at runtime so
the installed `libneml2_base.so` stays portable.

## Tests

```bash
# Python native tests.
pytest -q python/tests/native/

# Single file.
pytest -q python/tests/native/test_finite_volume.py

# C++ AOTI integration tests (excludes hidden benchmarks).
./build/aoti/tests/aoti/aoti_tests "[aoti]~[.bench]"

# A single C++ case (e.g. the sub-batch FV chain).
./build/aoti/tests/aoti/aoti_tests "[aoti_mode/composed]"
```

## Benchmarks

Source: `tests/aoti/bench_AOTIModel.cxx` (Catch2 `[.bench]`). All cases time
`Model::value` (or `Model::dvalue`) on **both** sides — the C++ figure and
the AOTI figure both include Model-class overhead (ValueMap pack/unpack,
dispatcher entry/exit, variable-store lookup), since both go through the
Factory. That's the speedup a user actually pays when they swap
`aoti_mode = true` into their `.i` file.

```bash
./build/aoti/tests/aoti/aoti_tests "[.bench]"                       # CPU all
CUDA_HOME=/usr/local/cuda-12.4 \
    ./build/aoti/tests/aoti/aoti_tests "[.bench]" --devices cuda    # +CUDA
./build/aoti/tests/aoti/aoti_tests "[bench/j2_radial]"              # one case
```

### `[bench/elasticity]` — forward `LinearIsotropicElasticity`

One forward segment, one `.pt2`. Tests the AOTI per-call overhead floor on a
constitutive law cheap enough that compute doesn't amortise framework cost.

CPU, float64, median of 100 runs after 10 warmup:

| batch | C++ `Model::value` (us) | AOTI `Model::value` (us) | speedup |
|---:|---:|---:|---:|
| 1 | 153.26 | 96.84 | **1.58x** |
| 32 | 165.55 | 110.11 | **1.50x** |
| 1,024 | 290.73 | 215.35 | **1.35x** |
| 65,536 | 3,110.64 | 2,870.39 | **1.08x** |

CUDA, RTX A5000, float64, median of 100 runs after 10 warmup:

| batch | C++ `Model::value` (us) | AOTI `Model::value` (us) | speedup |
|---:|---:|---:|---:|
| 1 | 523.25 | 407.00 | **1.29x** |
| 32 | 526.22 | 407.45 | **1.29x** |
| 1,024 | 551.79 | 409.31 | **1.35x** |
| 65,536 | 684.76 | 505.78 | **1.35x** |

### `[bench/j2_implicit]` — 8-D backward-Euler J2 return-map

One implicit segment: `_rhs.pt2` + `_step.pt2` + `_predictor.pt2`. Tests the
Newton-loop fusion (assemble + 8×8 LU + update + post-residual fused into one
AOTI graph per Newton iteration).

CPU, float64, median of 100 runs after 10 warmup:

| batch | C++ `Model::value` (us) | AOTI `Model::value` (us) | speedup |
|---:|---:|---:|---:|
| 1 | 14,353.89 | 703.72 | **20.40x** |
| 32 | 15,347.62 | 907.82 | **16.91x** |
| 1,024 | 28,280.97 | 4,902.98 | **5.77x** |
| 65,536 | 373,100.77 | 376,387.50 | **0.99x** |

CUDA, RTX A5000, float64, median of 100 runs after 10 warmup:

| batch | C++ `Model::value` (us) | AOTI `Model::value` (us) | speedup |
|---:|---:|---:|---:|
| 1 | 35,767.27 | 2,922.72 | **12.24x** |
| 32 | 36,255.32 | 2,860.95 | **12.67x** |
| 1,024 | 37,132.68 | 2,881.73 | **12.89x** |
| 65,536 | 82,540.11 | 23,084.80 | **3.58x** |

### `[bench/j2_radial]` — 3-segment radial-return J2

Same constitutive answer as `[bench/j2_implicit]`, different solve structure:
trial-state forward segment → 1-D Newton on `flow_rate` (vs 8-D in the
backward-Euler formulation) → explicit Euler post-update + stress. Three
segments orchestrated by `AOTIModel`'s segment loop, with separate `.pt2`s
per segment.

CPU, float64, median of 100 runs after 10 warmup:

| batch | C++ `Model::value` (us) | AOTI `Model::value` (us) | speedup |
|---:|---:|---:|---:|
| 1 | 10,459.30 | 679.40 | **15.39x** |
| 32 | 10,684.78 | 723.70 | **14.76x** |
| 1,024 | 15,902.91 | 1,847.18 | **8.61x** |
| 65,536 | 123,230.11 | 61,613.59 | **2.00x** |

CUDA, RTX A5000, float64, median of 100 runs after 10 warmup:

| batch | C++ `Model::value` (us) | AOTI `Model::value` (us) | speedup |
|---:|---:|---:|---:|
| 1 | 26,527.98 | 2,751.12 | **9.64x** |
| 32 | 26,447.72 | 2,674.16 | **9.89x** |
| 1,024 | 27,346.02 | 2,690.07 | **10.17x** |
| 65,536 | 36,429.39 | 7,733.13 | **4.71x** |

Phase-7 segment-composition net gain, vs the backward-Euler `[bench/j2_implicit]`
above:

| batch | CPU implicit / radial (us) | gain | CUDA implicit / radial (us) | gain |
|---:|---:|---:|---:|---:|
| 1 | 704 / 679 | **1.04x** | 2,923 / 2,751 | **1.06x** |
| 32 | 908 / 724 | **1.25x** | 2,861 / 2,674 | **1.07x** |
| 1,024 | 4,903 / 1,847 | **2.65x** | 2,882 / 2,690 | **1.07x** |
| 65,536 | 376,388 / 61,614 | **6.11x** | 23,085 / 7,733 | **2.98x** |

### `[bench/cp_single_crystal]` — single-crystal coupled crystal plasticity

Phase-10.13 Python-native CP leaves through the BLOCK-AOTI pipeline. Three
unknowns (``elastic_strain``, ``slip_hardening``, ``orientation``) in a single
per-crystal block with ``DenseLU``; predictor is
``CrystalPlasticityStrainPredictor``. The full residual chain (Rotation matrix
→ resolved shear → slip rule → hardening → BackwardEuler / WR2 exponential
integrators) is compiled into one ``_rhs.pt2``, one ``_step.pt2``, one
``_ift.pt2``, and one ``_predictor.pt2``.

CPU, float64, median of 100 runs after 10 warmup:

| batch | C++ `Model::value` (us) | AOTI `Model::value` (us) | speedup |
|---:|---:|---:|---:|
| 1 | 11,359.45 | 1,233.98 | **9.21x** |
| 32 | 15,364.53 | 2,960.04 | **5.19x** |
| 1,024 | 48,984.09 | 28,036.75 | **1.75x** |
| 65,536 | 1,440,487.71 | 2,229,360.63 | **0.65x** |

CUDA, RTX A5000, float64, median of 100 runs after 10 warmup:

| batch | C++ `Model::value` (us) | AOTI `Model::value` (us) | speedup |
|---:|---:|---:|---:|
| 1 | 27,998.32 | 4,449.90 | **6.29x** |
| 32 | 28,559.04 | 4,410.19 | **6.48x** |
| 1,024 | 31,114.38 | 9,594.32 | **3.24x** |
| 65,536 | 247,874.17 | 432,784.58 | **0.57x** |

The pattern matches the J2 implicit benches — large speedup at small batch
(framework overhead amortised by AOTI's fused Newton loop) tapering toward
break-even as the per-iteration compute dominates. The >1M-element ``65,536``
row crosses below 1× because the linear-solve overhead Inductor's
``aten._linalg_solve_ex.default`` falls back to ``proxy_executor`` for (see
the compile-time warning) inflates per-iter cost; this is upstream and will
lift when LinAlg ops gain a native c-shim.

Numbers above predate the D-062 typed-JVP rewrite (they were measured when
CP's per-slip rotation derivative was the closed-form ``d_RXR_T_*_dR``
matmul broadcast, since replaced by the typed product-rule primitives
``jvp_rotate_sym`` / ``jvp_rotate_skew``); they still characterize the graph
shape, which is dominated by the per-slip vectorization (no Python loops over
slip systems at trace time — see ``models/crystal_plasticity.py`` and
``types/functions.py`` for the C++-style batched algebra). The
prior loop-form numbers (CPU B=1 ``1,477.27 us → 7.62x``, CUDA B=1
``5,004.34 us → 5.31x``) traced a ~3,200-node step graph; vectorization
cut that to ~1,600 nodes and lifted small-batch speedup by ~20 % on both
devices.

#### Why the CP step graph is still ~2.5× bigger than j2_implicit's

After all the per-slip / basis-inject vectorisation, the rhs graph (300
nodes) is within ~2× of ``j2_implicit_rhs`` (161 nodes) — which is the
right ballpark given CP has more leaves (RotationMatrix, ResolvedShear,
PlasticDeformationRate, PlasticVorticity, OrientationRate, three time
integrators) and an extra unknown axis (Rot=3 on top of SR2=6 and
Scalar=1). The step / IFT graphs stay at ~2.5× because the
**chain-rule pushforward machinery** is genuinely heavier: cross-type
blocks like ``∂SR2/∂Rot`` flow through the product-rule JVP primitives
(``jvp_rotate_sym`` / ``jvp_rotate_skew``), and CP has several of those
that j2's all-symmetric chain doesn't.

A historical experiment (on the since-removed ``d_RXR_T_*`` matmul
kernels, and still applicable to the surviving pack/unpack helpers
``mandel_pack_sym3``, ``skew_pack_axial3``, ``r2_from_sr2``,
``r2_from_wr2``, ``sym``, ``skew``) collapsed their inner Mandel-slot
loops into single ``flat @ projection``
matmuls. Graph nodes dropped a further ~30 %, **CPU wall-time improved
~10 %**, but **CUDA wall-time regressed ~9 %** across the batch sweep.
Inductor doesn't fuse a separate matmul kernel with surrounding
pointwise ops, while the explicit ``select+stack`` chains fuse into a
single Triton kernel via the pointwise scheduler. The CUDA loss at small
batch outweighed the CPU win, so the matmul-against-fixed-projection
form was reverted for these helpers — comments on the affected functions
document the empirical reasoning. The same caveat applies to every
free function in ``types/functions.py`` that operates on a small,
statically-known base shape (e.g. ``deuler_rodrigues``,
``exp_map``/``dexp_map``): replacing the closed-form pointwise body with
a constant-projection matmul is unlikely to be a net win on CUDA at
small batch without changes to the kernel fusion path.

Future-work candidate: a hand-rolled ``unrolled_matmul(A, B)`` for tiny
contraction dimensions that emits the equivalent pointwise ops directly
(``[A[..., i, 0] * B[..., 0, j] + A[..., i, 1] * B[..., 1, j] + ...
for i, j in (n_row, n_col)]``) instead of dispatching to BLAS / Triton's
matmul. That would in principle unify the CPU and CUDA wins — Inductor
would fuse the resulting op chain on both devices. Worth checking
whether Inductor's own "tiny matmul as pointwise" heuristic already
does this for newer versions before hand-rolling.

### `[bench/cp_taylor]` — Taylor crystal plasticity (BLOCK + DENSE Schur)

Phase-10.9 BLOCK + DENSE BlockNewtonStep through the Block-AOTI pipeline.
Per-crystal block (``elastic_strain``, ``slip_hardening``, ``orientation``,
sub-batched over ``ncrystal``) reconciled with a global DENSE block
(``deformation_rate``, ``target_cauchy_stress``) through
:class:`~neml2.solvers.SchurComplement` — primary solver handles the
batched per-crystal ``(50, 50)`` block via ``torch.linalg.solve``; Schur
solver handles the reduced ``(12, 12)`` DENSE complement.

Unlike the other CP fixture, the meaningful scale axis for a Taylor
aggregate is the grain count (a typical polycrystal is one material point
with hundreds to thousands of grains), so the bench fixes ``B=1`` and
sweeps ``ncrystal``. Wrappers ``cp_taylor_{native,cxx}_N{N}.i`` shape-
specialize the per-crystal axis at AOTI export.

float64, median of 200/ncrystal runs (with 1/5 warmup):

| ncrystal | CPU cxx (us) | CPU AOTI (us) | CPU speedup | CUDA cxx (us) | CUDA AOTI (us) | CUDA speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | 145,268 | 7,745     | **18.76x** | 338,367 | 21,206  | **15.96x** |
| 10  | 657,106 | 265,296   | **2.48x**  | 330,196 | 26,704  | **12.37x** |
| 50  | 218,656 | 514,572   | **0.42x**  | 333,879 | 90,477  | **3.69x**  |
| 100 | 261,345 | 1,213,905 | **0.22x**  | 335,689 | 225,634 | **1.49x**  |

CUDA holds a useful speedup all the way to ``ncrystal = 100`` because the
per-crystal block solves go through a single batched
``torch.linalg.solve`` whose GPU kernels amortize launch overhead across
the grain dim. CPU AOTI loses by ``ncrystal = 50`` and is 5× slower at
``ncrystal = 100``: the same ``aten._linalg_solve_ex.default``
``proxy_executor`` fallback that hurts the j2 64k row hurts each Newton
iter here, plus the chain-rule block-matrix assembly inside
``BlockIFT._assembled_matrix`` does work the C++ eager path doesn't.
Lifts when LinAlg ops get a native c-shim (same upstream gate as the
single-crystal 64k row).

The ``ncrystal=1`` row is a useful cross-check on the BLOCK + DENSE
chain rule's size-1 sub-batch handling: it stress-tests the path through
``_expanded_identity_seed`` whose ``sb_total == 1`` fast-path used to
emit a tangent that claimed ``sub_batch_ndim = 1`` while dropping the
size-1 sub-batch axis from the data shape — downstream
the old sub-batch padding helper then inserted the axis at the wrong slot and the
chain-rule block-matrix broadcast collapsed at AOTI trace time. Fixed
by materialising every sub-batch axis explicitly (even the size-1 ones)
in the early-return path. The dedicated ``cp_single_crystal`` bench
above still applies for pure single-crystal use (no BLOCK + DENSE
partition needed).

### `Model::dvalue` benchmarks (Phase 9 cross-segment composition)

Same fixtures as the value benches above, run through `Model::dvalue`. On the
C++ side that's the eager analytical chain-rule machinery; on the AOTI side
that's the cross-segment composition (per-segment `_jvp.pt2` / `_ift.pt2`
loads + column-stack `dstate` matmul + per-(out, in) unpack into the standard
derivative storage).

```bash
./build/aoti/tests/aoti/aoti_tests \
    "[bench/elasticity_dvalue],[bench/j2_implicit_dvalue],[bench/j2_radial_dvalue]"
```

`[bench/elasticity_dvalue]` — forward elasticity SSR4:

| batch | CPU cxx / aoti (us) | gain | CUDA cxx / aoti (us) | gain |
|---:|---:|---:|---:|---:|
| 1 | 143 / 169 | **0.85x** | 476 / 570 | **0.84x** |
| 32 | 149 / 210 | **0.71x** | 477 / 569 | **0.84x** |
| 1,024 | 204 / 1,147 | **0.18x** | 478 / 586 | **0.82x** |
| 65,536 | 1,573 / 34,852 | **0.05x** | 516 / 2,421 | **0.21x** |

`[bench/j2_implicit_dvalue]` — 8-D backward-Euler J2 IFT:

| batch | CPU cxx / aoti (us) | gain | CUDA cxx / aoti (us) | gain |
|---:|---:|---:|---:|---:|
| 1 | 16,468 / 1,367 | **12.04x** | 42,756 / 5,115 | **8.36x** |
| 32 | 17,787 / 1,764 | **10.09x** | 43,332 / 4,950 | **8.75x** |
| 1,024 | 34,970 / 8,250 | **4.24x** | 43,496 / 5,016 | **8.67x** |
| 65,536 | 535,017 / 752,298 | **0.71x** | 104,168 / 38,855 | **2.68x** |

`[bench/j2_radial_dvalue]` — 3-segment radial-return composition:

| batch | CPU cxx / aoti (us) | gain | CUDA cxx / aoti (us) | gain |
|---:|---:|---:|---:|---:|
| 1 | 17,352 / 1,410 | **12.31x** | 48,833 / 5,131 | **9.52x** |
| 32 | 18,358 / 1,781 | **10.31x** | 49,057 / 5,150 | **9.53x** |
| 1,024 | 36,287 / 8,278 | **4.38x** | 49,333 / 5,145 | **9.59x** |
| 65,536 | 379,022 / 628,070 | **0.60x** | 94,270 / 35,517 | **2.65x** |
