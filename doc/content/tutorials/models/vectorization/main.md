---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

(tutorials-models-vectorization)=
# Vectorization

Every NEML2 model is *inherently batched*. The forward operator
processes a leading batch of states in a single call — the same code
path that handles one input handles ten, ten thousand, or a multi-axis
grid of inputs. From Python this means you almost never want a
`for`-loop around `model(...)`: build the input as a single tensor with
a leading batch dimension, call the model once, get a batched output
back.

This tutorial shows the mechanics on the elasticity model from
[](tutorials-models-running-your-first-model), then puts the loop and
the batched call side-by-side on the clock so the cost of getting it
wrong is concrete.

## Why vectorization

Modern CPUs and GPUs hit their peak throughput only when the same
operation is applied to a contiguous chunk of data at once — SIMD on
the CPU, SIMT on CUDA. NEML2 lays out every variable, parameter, and
buffer contiguously and dispatches one PyTorch kernel per model
operation, so a batched call lets the kernel work at hardware peak.
A Python loop instead pays the per-call dispatch overhead — Python
interpreter, autograd bookkeeping, kernel launch — *once per item*,
which dominates the actual arithmetic for any non-trivial batch size.

## The input file

We reuse the elasticity model from the first tutorial unchanged:

```{literalinclude} input.i
:language: ini
:caption: input.i
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
strain1 = SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
strain1.data.shape, model(strain1).data.shape
```

## A batch of states in one call

To evaluate the model on `N` strain states at once, build the input
with a *leading batch dimension*: shape `(N, 6)` instead of `(6,)`.
NEML2 detects the extra leading dim and broadcasts every internal
operation across it. The output comes back with the same leading
shape:

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
loop required. The leading dimension is free-form: you can use a 2-D
batch like `(n_load_steps, n_samples)` and the model returns
`(n_load_steps, n_samples, 6)`, and so on.

```{code-cell} ipython3
multi = torch.zeros(4, 3, 6)
multi[..., 0] = torch.linspace(0.0, 0.01, 4).unsqueeze(-1)
model(SR2(multi)).data.shape
```

## Loop vs. batched: a numerical experiment

How much does the loop actually cost? We evaluate the same elastic
model on $N=10{,}000$ strain states two ways — once as a Python loop
over single states, once as a single batched call — and time just the
model evaluations (input allocation is hoisted out so it isn't
counted):

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

print(f"python loop: {t_loop*1e3:8.2f} ms")
print(f"batched:     {t_batch*1e3:8.2f} ms")
print(f"speedup:     {t_loop/t_batch:8.1f}x")
```

Two orders of magnitude is typical even for this trivial model on
CPU; on GPU the gap widens further because kernel-launch overhead is
larger relative to the actual compute. For larger models the
absolute numbers grow but the *ratio* stays in the same ballpark —
the loop is paying constant per-call overhead that the batched call
amortizes across the whole batch.

## Dynamic vs. sub-batch dimensions

The leading shape we used above is what NEML2 calls a **dynamic batch
dimension** — it's free-form, sized at call time, and the model has no
opinion about what's in it. That's the common case and what you want
99% of the time.

Some advanced contexts (e.g., the residual axis of an
`ImplicitUpdate`, or a parameter that itself carries a sample axis)
introduce a **sub-batch dimension** — a fixed-size axis that's part of
a tensor's logical shape rather than a free-form batch. The
`TensorWrapper.sub_batch_ndim` field tracks it, and broadcasting
rules treat sub-batch axes differently from dynamic ones. The
distinction rarely matters for the everyday "feed in a batch, get a
batch back" pattern, but it's worth knowing the term exists.

## Where to go next

- [](tutorials-models-evaluation-device) — the same batched call runs
  on CUDA the moment you move the model and the input there with
  `.to(device)`, which is where the SIMT story really pays off.
- [](tutorials-models-cross-referencing) and
  [](tutorials-models-composition) — once a model is composed of
  several pieces, the same batched-call semantics propagate through
  every internal evaluation.
- [](tutorials-models-transient-driver) — for time-stepping a batched
  state through a load history, where each batched call is one step.
