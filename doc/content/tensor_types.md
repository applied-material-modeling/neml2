(tensor-types)=
# Tensor types

NEML2 evaluates constitutive models on batched tensor data. The tensor
backend is PyTorch — every value in the system is a `torch.Tensor` at
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
— "everything that isn't base". `wrapper.batch_ndim`, `.dynamic_ndim`,
and `.sub_batch_ndim` provide the corresponding counts.

## Fixed-base-shape tensor types

The following wrappers ship in `neml2.types`. All inherit from
`TensorWrapper` and are `@dataclass(frozen=True, eq=False)` with the
same two storage fields (`data`, `sub_batch_ndim`) plus the class-level
`BASE_NDIM` / `BASE_SHAPE` invariants.

| Type           | Base shape | Storage / convention                                                                                                  |
| :------------- | :--------- | :-------------------------------------------------------------------------------------------------------------------- |
| `Scalar`       | `()`       | A single number per batch entry. Wrapped (instead of being raw `torch.Tensor`) so reflected-operator dispatch like `Scalar * SR2 → SR2` is unambiguous. |
| `Vec`          | `(3,)`     | 3-vector.                                                                                                              |
| `Rot`          | `(3,)`     | Rodrigues vector representing a 3D rotation.                                                                          |
| `MillerIndex`  | `(3,)`     | Integer-coordinate crystallographic direction or plane normal, stored as float for differentiability.                 |
| `R2`           | `(3, 3)`   | Full second-order tensor (no symmetry).                                                                                |
| `WR2`          | `(3,)`     | Skew-symmetric second-order tensor in *Mandel-style* axial-vector packing — the three independent components stored as a 3-vector. |
| `SR2`          | `(6,)`     | Symmetric second-order tensor in **Mandel notation**: `(σ₁₁, σ₂₂, σ₃₃, √2·σ₂₃, √2·σ₁₃, √2·σ₁₂)`. The off-diagonal `√2` factors make the inner product `A : B = a · b` in 6-vector form. |
| `SSR4`         | `(6, 6)`   | Fourth-order tensor with both pairs of minor symmetries (the symmetry class of the elasticity tensor `ℂ`) in Mandel packing. |

All wrappers default to `torch.float64`. Construct them either from a
raw `torch.Tensor` (`SR2(torch.tensor([0.01, 0, 0, 0, 0, 0]))`) or, for
`Scalar`, directly from a Python number (`Scalar(200e3)` — defaults to
`float64`).

## Batching

A model's forward operator dispatches one PyTorch kernel per
mathematical operation. Those kernels broadcast across leading batch
axes natively, so the same `forward(x)` body handles a single state
and a multi-million-state batch with no rewrites — the only difference
is the shape of the input tensors.

The canonical pattern:

```python
strain_single = SR2(torch.tensor([0.01, 0, 0, 0, 0, 0]))  # base only
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
single-state model call is almost always a mistake — see
[](tutorials-models-vectorization) for the cost.

## Broadcasting

Inside `forward`, binary operators between typed wrappers broadcast
their batch regions using PyTorch's standard rules:

- Right-aligned by axis (so a `(6,)` SR2 and a `(N, 6)` SR2 broadcast
  to `(N, 6)`).
- Size-1 axes are stretched to match.
- Mismatched non-1 sizes are an error.

Algebraic operators preserve sub-batch metadata: every binary op
routes through `align_sub_batch` (see `neml2/types/_base.py`), which
unifies the two operands' sub-batch widths and tags the result with
the correct `sub_batch_ndim`. The upshot is that a "global" `Scalar`
parameter and a "per-site" `Scalar` field combine cleanly at any
dynamic batch size.

## Dynamic vs sub-batch dimensions

These are the two batching regions the framework treats differently:

| Region        | Sized at... | Traced as...                | Broadcasts with...     | Typical use                                              |
| :------------ | :---------- | :-------------------------- | :--------------------- | :------------------------------------------------------- |
| Dynamic batch | call time    | dynamic dim (`Dim("batch")`) | everything             | every "ordinary" batch — N material points, time steps. |
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

T_controls = Scalar(
    torch.linspace(300.0, 1200.0, 20, dtype=torch.float64)
).sub_batch.retag(1)
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
    expr = 'Scalar(torch.linspace(300.0, 1200.0, 20, dtype=torch.float64)).sub_batch.retag(1)'
  []
[]
```

## Construction surface

Beyond raw-tensor construction, the wrappers ship a few convenience
constructors used throughout the docs and the test suite:

- `Scalar(<float>)` / `Scalar(<int>)` — promote a Python number
  directly to a 0-dim `Scalar` with `torch.float64`.
- `SR2.fill(a11, a22, a33, a23, a13, a12)` — build an SR2 from its six
  Mandel components (with the `√2` factors handled internally).
- `SR2.zeros()`, `Vec.zeros()`, etc. — zero-valued wrappers.
- `<Wrapper>.from_value(x, like=other_wrapper)` — promote a Python
  literal inheriting the `dtype` and `device` of an existing wrapper
  (used heavily inside leaf `forward` implementations to build
  in-place neutrals).

Every constructor accepts an optional `device=` / `dtype=` kwarg where
applicable; see [](tutorials-models-evaluation-device) for the device
story.

## See also

- [](tutorials-models-vectorization) — the everyday user-facing view
  of batching with a timed loop-vs-batched comparison.
- [](tutorials-models-evaluation-device) — moving wrappers across
  devices with `.to(device=...)`.
- [](tutorials-models-input-file) — the `[Tensors]` section that
  constructs wrappers from HIT.
- `neml2/types/_base.py` — the canonical implementation reference for
  the shape decomposition + `align_sub_batch` machinery.
