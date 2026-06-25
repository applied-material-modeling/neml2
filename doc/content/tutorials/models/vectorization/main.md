---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(tutorials-models-vectorization)=
# Vectorization

You'll evaluate the elasticity model from
[](tutorials-models-running-your-first-model) on a whole batch of
strain states in one call, then time it against a Python `for`-loop
to see how much the loop costs. The short version: when you have
many states, stack them into one batched input and let the model
evaluate them in a single call.

```{code-cell} ipython3
:tags: [remove-cell]

# When this notebook runs in Google Colab, install NEML2 from PyPI. The guard
# makes the cell a no-op everywhere else (the docs build and local Jupyter
# already have NEML2 installed), and the cell is hidden from the rendered docs.
import sys

if "google.colab" in sys.modules:
    !pip install -q neml2
```

## The input file

We reuse the elasticity model from the first tutorial unchanged:

```{code-cell} ipython3
%%writefile input.i
# Reuses the hello-world elasticity model from the
# "Running your first model" tutorial. Linear isotropic elasticity
# is a trivial map (SR2 -> SR2), which keeps the focus on *how* we
# feed in a batch of strains rather than what the model is doing.
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

```{code-cell} ipython3
import torch
import neml2
from neml2.types import SR2

torch.set_default_dtype(torch.float64)

model = neml2.load_model("input.i", "elasticity")
model
```

## One state at a time

A single (unbatched) strain has base shape `(6,)` — the six independent
components of an `SR2` in Mandel packing. The model returns a stress
of the same shape:

```{code-cell} ipython3
strain1 = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
strain1.data.shape, model(strain1).data.shape
```

## A batch of states in one call

To evaluate the model on `N` strain states at once, build the input
with a *leading batch dimension*: shape `(N, 6)` instead of `(6,)`.
The output comes back with the same leading shape:

```{code-cell} ipython3
N = 5
batch_data = torch.zeros(N, 6)
batch_data[:, 0] = torch.linspace(0.0, 0.01, N)  # ramp epsilon_xx from 0 to 1%
strains = SR2(batch_data)

stresses = model(strains)
strains.data.shape, stresses.data.shape
```

```{code-cell} ipython3
stresses.data
```

Each row of the output is the stress for the corresponding strain — no
loop required. The leading dimension is free-form: a 2-D batch like
`(n_load_steps, n_samples)` returns `(n_load_steps, n_samples, 6)`,
and so on.

```{code-cell} ipython3
multi = torch.zeros(4, 3, 6)
multi[..., 0] = torch.linspace(0.0, 0.01, 4).unsqueeze(-1)
model(SR2(multi)).data.shape
```

## Loop vs. batched: a numerical experiment

How much does the loop actually cost? Evaluate the same elastic model
on $N=10{,}000$ strain states two ways — a Python loop over single
states and one batched call — and time just the model evaluations:

```{code-cell} ipython3
import time

N = 10_000
batch_data = torch.zeros(N, 6)
batch_data[:, 0] = torch.linspace(0.0, 0.01, N)
strains = SR2(batch_data)

# --- Python loop: one model call per state ---
t0 = time.perf_counter()
for i in range(N):
    model(SR2(batch_data[i]))
t_loop = time.perf_counter() - t0

# --- Single batched call: one model invocation, N states ---
t0 = time.perf_counter()
model(strains)
t_batch = time.perf_counter() - t0

print(f"python loop: {t_loop * 1e3:8.2f} ms")
print(f"batched:     {t_batch * 1e3:8.2f} ms")
print(f"speedup:     {t_loop / t_batch:8.1f}x")
```

The cell prints the speedup — large even for a model this trivial
on CPU, because the loop pays a constant per-call overhead that the
batched call amortizes across the whole batch. On GPU, where each
Python-level launch carries extra latency, the gap is usually wider
still.

## Where to go next

- [](tutorials-models-evaluation-device) — the same batched call
  runs on CUDA once you move the model and the input there with
  `.to(device)`; on GPU, the per-call overhead the batched form
  avoids is usually even more pronounced.
- [](tutorials-models-cross-referencing) and
  [](tutorials-models-composition) — once a model is composed of
  several pieces, the same batched-call semantics propagate through
  every internal evaluation.
- [](tutorials-models-transient-driver) — for time-stepping a batched
  state through a load history, where each batched call is one step.
