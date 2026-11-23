(tensor-types)=
# Tensor types

NEML2 evaluates constitutive models on batched tensor data, and PyTorch
is the tensor backend — every value in the system is a `torch.Tensor` at
the storage level. But a bare `torch.Tensor` is just a block of numbers
with a shape: nothing in `torch.randn(6)` says whether those six numbers
are a stress, a strain, or six unrelated scalars, and nothing protects
the `√2` packing convention a symmetric tensor relies on. NEML2 closes
that gap by wrapping each tensor in a *typed wrapper*.

## What a wrapper is

Each wrapper is a small frozen dataclass holding **one** field of
substance — `data: torch.Tensor` — plus a little metadata. The wrapped
tensor is always reachable as `.data`:

```python
import torch
from neml2.types import SR2

stress = SR2(torch.tensor([100.0, 50.0, 50.0, 0.0, 0.0, 0.0]))
stress.data            # the underlying torch.Tensor
stress.data.shape      # torch.Size([6])
```

The wrapper adds two things on top of the raw storage:

- **A fixed mathematical structure.** `SR2` *is* a symmetric
  second-order tensor: its trailing axes have a known shape (`(6,)`)
  and a known packing convention (Mandel notation), so `SR2` arithmetic,
  invariants, and conversions all mean the right thing. The wrapper type
  is what makes `Scalar * SR2 → SR2` dispatch correctly and what lets a
  function declare "I take an `SR2` and return a `Scalar`".
- **Batching metadata.** Beyond the fixed base axes, a wrapper records
  how its remaining axes are split into an ordinary *dynamic* batch and
  an optional *sub-batch* region for per-site structure. This is the
  part that has no equivalent on a raw `torch.Tensor`.

## How to read this section

If you already know PyTorch tensors, the fastest way to learn the
wrappers is to focus on **where they differ** from a raw `torch.Tensor`
— that is how the rest of this section is organized:

- [](tensor-types-primitive) — the catalog of fixed-base-shape types
  (`Scalar`, `Vec`, `SR2`, …), their packing conventions, and how to
  construct them.
- [](tensor-types-regions) — how a wrapper's shape splits into base,
  sub-batch, and dynamic-batch regions, and why the split matters.
- [](tensor-types-batching) — vectorizing a model over a batch, the
  broadcasting rules, and the sub-batch region that torch has no notion
  of.
- [](tensor-types-region-views) — the region-view properties that make
  every reshape, reduction, and concatenation name the region it acts
  on (and disambiguate the free functions).
- [](tensor-types-indexing) — region-scoped indexing and slicing, and
  how it relates to ordinary NumPy/torch indexing.

```{toctree}
:maxdepth: 1
:hidden:

primitive_types
regions
batching
region_views
indexing
```

## See also

- [](tutorials-models-vectorization) — the everyday user-facing view
  of batching with a timed loop-vs-batched comparison.
- [](tutorials-models-evaluation-device) — moving wrappers across
  devices with `.to(device=...)`.
- [](tutorials-models-input-file) — the `[Tensors]` section that
  constructs wrappers from HIT.
