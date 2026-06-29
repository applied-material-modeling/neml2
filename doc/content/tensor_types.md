(tensor-types)=
# Tensor types

NEML2 evaluates constitutive models on batched tensor data. The tensor
backend is PyTorch — every tensor in the system is a `torch.Tensor` at
the storage level — but NEML2 wraps each tensor in a *typed wrapper*
that carries a fixed mathematical structure (a scalar, a vector, a
symmetric second-order tensor, …) and a small amount of batching
metadata.

This page documents the wrappers, the shape conventions, and the
batching rules. It is the reference complement to
[](tutorials-models-vectorization), which works through a single
worked example.

## Shape decomposition

Every typed wrapper exposes its underlying tensor as the `.data`
attribute. The shape of that tensor splits into three contiguous
regions:

```
data.shape == (*dynamic_batch_shape, *sub_batch_shape, *base_shape)
              └── leading ──┘└── middle (static) ──┘└─ BASE_NDIM ─┘
```

- **Base shape** — the trailing `BASE_NDIM` axes. Fixed by the wrapper
  type (e.g. `(6,)` for `SR2`, `()` for `Scalar`); these encode the
  mathematical structure and never participate in broadcasting.
- **Sub-batch shape** — the next `sub_batch_ndim` axes. A small,
  *static* batching region used to express per-site structure
  (lookup-table axes, finite-volume cells, slip systems, …). The
  chain-rule machinery treats sub-batch dims specially so derivatives
  stay consistent when models on different sites are composed.
  Default 0 — most models don't need it.
- **Dynamic batch shape** — everything left over. Free-form, sized at
  call time, traced as dynamic by `torch.export` so a single AOTI
  artifact handles every batch size from 1 to roughly a million
  without recompilation.

`wrapper.batch_shape` returns `dynamic_batch_shape + sub_batch_shape`
— "everything that isn't base". `wrapper.batch.ndim`,
`wrapper.dynamic_batch.ndim`, and `wrapper.sub_batch_ndim` provide the
corresponding counts.

## Fixed-base-shape tensor types

The following wrappers ship in `neml2.types`. The class hierarchy is:

```
TensorWrapper           (abstract — shape decomposition + region views)
    └── PrimitiveTensor (concrete intermediate — generic ops + factories)
            ├── Scalar
            ├── Vec, R2, SR2, WR2, MRP, SSR4, MillerIndex
```

`PrimitiveTensor` is the layer where the generic arithmetic operators
(`+`, `-`, `*`, `/`, `-x`) and shape factories (`zeros`, `ones`, `full`,
`empty`, `fill`) are defined. Each concrete leaf below it adds any
class-specific factories — e.g. `R2.identity`, `SSR4.identity_sym`,
`SR2.fill` with Mandel √2 scaling.

| Type           | Base shape | Storage / convention                                                                                                  |
| :------------- | :--------- | :-------------------------------------------------------------------------------------------------------------------- |
| `Scalar`       | `()`       | A single number per batch entry. The wrapper exists so mixed operations like `Scalar * SR2` reliably return an `SR2`. |
| `Vec`          | `(3,)`     | 3-vector.                                                                                                              |
| `MRP`          | `(3,)`     | Modified Rodrigues parameters (MRPs) representing a 3D rotation: `n * tan(θ/4)`, zero vector = identity.             |
| `MillerIndex`  | `(3,)`     | Integer-coordinate crystallographic direction or plane normal, stored as float for differentiability.                 |
| `R2`           | `(3, 3)`   | Full second-order tensor (no symmetry).                                                                                |
| `WR2`          | `(3,)`     | Skew-symmetric second-order tensor stored as an axial 3-vector `(w0, w1, w2)`. No √2 scaling; the corresponding 3×3 skew form is recovered by `r2_from_wr2`. |
| `SR2`          | `(6,)`     | Symmetric second-order tensor in **Mandel notation**: `(σ₁₁, σ₂₂, σ₃₃, √2·σ₂₃, √2·σ₁₃, √2·σ₁₂)`. The off-diagonal `√2` factors make the inner product `A : B = a · b` in 6-vector form. |
| `SSR4`         | `(6, 6)`   | Fourth-order tensor with both pairs of minor symmetries (the symmetry class of the elasticity tensor `ℂ`) in Mandel packing. |

`Scalar` defaults to `torch.float64` for both Python-literal construction
and its factory methods; the other wrappers accept a `dtype=` kwarg and
fall through to torch's global default otherwise. Construct them from a
raw `torch.Tensor` (`SR2.fill(0.01, 0, 0, 0, 0, 0)`), from a
Python literal (`Scalar(200e3)`), or via the inherited factories
(`Vec.zeros(N)`, `SR2.fill(σ11, σ22, σ33, σ23, σ13, σ12)`, etc.).

## Batching

A model's forward operator dispatches one PyTorch kernel per
mathematical operation. Those kernels broadcast across leading batch
axes natively, so the same `forward(x)` body handles a single state
and a multi-million-state batch with no rewrites — the only difference
is the shape of the input tensors.

The canonical pattern:

```python
strain_single = SR2.fill(0.01, 0, 0, 0, 0, 0)  # base only
stress = model(strain_single)
# stress.data.shape == (6,)

strain_batch = SR2(torch.randn(10_000, 6) * 0.01)         # one leading batch dim
stress = model(strain_batch)
# stress.data.shape == (10_000, 6)

strain_grid = SR2(torch.randn(50, 200, 6) * 0.01)         # two leading batch dims
stress = model(strain_grid)
# stress.data.shape == (50, 200, 6)
```

Leading batch dims are completely free-form. A Python loop around a
single-state model call leaves a lot of throughput on the table —
pass the whole batch as one tensor whenever you can.
[](tutorials-models-vectorization) shows the timing difference.

## Broadcasting

Inside `forward`, binary operators between typed wrappers broadcast
their batch regions using PyTorch's standard rules:

- Right-aligned by axis (so a `(6,)` SR2 and a `(N, 6)` SR2 broadcast
  to `(N, 6)`).
- Size-1 axes are stretched to match.
- Mismatched non-1 sizes are an error.

Algebraic operators preserve sub-batch metadata: every binary op
unifies the two operands' sub-batch widths and tags the result
accordingly. The upshot is that a "global" `Scalar` parameter and a
"per-site" `Scalar` field combine cleanly at any dynamic batch size.

## Dynamic vs sub-batch dimensions

These are the two batching regions the framework treats differently:

| Region        | Sized at... | Traced as...                | Broadcasts with...     | Typical use                                              |
| :------------ | :---------- | :-------------------------- | :--------------------- | :------------------------------------------------------- |
| Dynamic batch | call time    | dynamic dim                  | everything             | every "ordinary" batch — N material points, time steps. |
| Sub-batch     | construction time | static shape          | other sub-batches of matching width | per-site structure — interpolation-table axis, FV cell index, slip-system axis. |

Operationally:

- Default `sub_batch_ndim = 0`. Models that don't need the distinction
  ignore it; everything sits in the dynamic region.
- Promote axes to sub-batch with `.sub_batch.retag(n)` (`n` = number of
  trailing batch axes to mark). The most common case is `n = 1`
  marking a lookup-table or per-cell axis.
- Sub-batch dims do NOT participate in dynamic-batch broadcasting.
  They behave like a small extra structural region the chain-rule
  machinery accumulates over.

A representative use, from the Kocks–Mecking shear-modulus lookup
table:

```python
from neml2.types import Scalar
import torch

T_controls = linspace(Scalar(300.0).sub_batch, Scalar(1200.0).sub_batch, 20)
```

This marks the trailing length-20 axis as the sub-batch (interpolation
control points), so any model consuming `T_controls` accumulates its
chain-rule contribution across that axis without conflating it with
the dynamic per-state batch.

`.sub_batch.retag(n)` is also accepted inside a `[Tensors]` HIT block:

```ini
[Tensors]
  [T_controls]
    type = Python
    expr = 'linspace(Scalar(300.0).sub_batch, Scalar(1200.0).sub_batch, 20)'
  []
[]
```

## Region views

Shape-manipulation methods live on four region-view properties so the
intent of any reshape, broadcast, or reduction is unambiguous:

- `t.batch` — the combined `dynamic_batch + sub_batch` region.
  Read-only `.shape` / `.ndim`; shape-changing ops raise so callers
  pick `dynamic_batch` or `sub_batch` explicitly. The free function
  `cat` in `neml2.types` accepts a batch view if you do need to
  concatenate across the combined region.
- `t.dynamic_batch` — dynamic batch only. Ops preserve `sub_batch_ndim`.
- `t.sub_batch` — sub-batch only. Ops adjust `sub_batch_ndim`.
- `t.base` — the base region. Read-only except for `transpose` on the
  square-base types (`R2`, `SSR4`).

Every mutable view exposes the same surface: `.shape`, `.ndim`,
`.unsqueeze(dim)`, `.squeeze(dim)`, `.expand(*shape)`. Concatenation
along a region axis goes through the free function `cat` in
`neml2.types` (see below).
The view methods return a fresh wrapper, so calls chain cleanly:

```python
broadcast = SR2.fill(0.1, -0.05, -0.05, 0, 0, 0).dynamic_batch.expand(20)
# Construct an SR2 of base shape (6,), then broadcast it to (20, 6).

retagged = linspace(Scalar(0).sub_batch, Scalar(1).sub_batch, 5)
# Mark the trailing length-5 axis as sub-batch.

tr_R = R.base.transpose(-2, -1)   # Transpose the (3, 3) base of an R2.
```

The companion free functions `sum`, `mean`, `diff` in `neml2.types`
take a view argument and dispatch on its kind:

```python
from neml2.types import sum, mean, diff

avg = mean(t.sub_batch, dim=0)        # Reduce sub-batch axis 0.
total = sum(t.dynamic_batch, [0, 1])  # Reduce two dynamic axes.
delta = diff(t.sub_batch, n=1, dim=-1)
```

## Construction surface

Beyond raw-tensor construction, every primitive inherits a small
factory family from `PrimitiveTensor`:

- `<T>.zeros(*batch, dtype=None, device=None)` — zero-filled wrapper of
  dynamic shape `batch` and base `T.BASE_SHAPE`.
- `<T>.ones(*batch, ...)`, `<T>.full(*batch, fill_value=..., ...)`,
  `<T>.empty(*batch, ...)`.
- `<T>.fill(*components, ...)` — reshape `prod(T.BASE_SHAPE)` scalars
  into the base. `SR2.fill` overrides this with Mandel-aware 1 / 3 / 6
  component overloads (the √2 shear scaling is internal).
- `<T>.identity(...)` where mathematically meaningful (`R2`,
  `SR2`, `WR2`, `MRP`, `SSR4`'s several projector variants).

`Scalar` adds the torch-analogue factories:

- `Scalar(<float>)` / `Scalar(<int>)` / `Scalar(<list>)` — direct
  literal coercion, defaults to `torch.float64`.
- `Scalar.zeros`, `Scalar.ones`, `Scalar.full` — override the
  `PrimitiveTensor` defaults to keep `float64`.
- `linspace(Scalar(start).dynamic_batch, Scalar(end).dynamic_batch, steps)`, `Scalar.arange(start, end, step)` —
  mirror the torch creation API.
- `Scalar.from_value(x, like=other_wrapper)` — promote a Python literal
  inheriting `dtype`/`device` from an existing wrapper. Useful inside
  leaf `forward` to build in-place neutrals.

Every constructor accepts an optional `device=` / `dtype=` kwarg; see
[](tutorials-models-evaluation-device) for the device story.

## See also

- [](tutorials-models-vectorization) — the everyday user-facing view
  of batching with a timed loop-vs-batched comparison.
- [](tutorials-models-evaluation-device) — moving wrappers across
  devices with `.to(device=...)`.
- [](tutorials-models-input-file) — the `[Tensors]` section that
  constructs wrappers from HIT.
