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

(tutorials-models-evaluation-device)=
# Evaluation device

You'll move a model and its inputs onto a target device (CPU here, but
the same calls work for CUDA or any other torch device) and run a
forward pass.

The runnable cells stay on CPU so the doc build is portable. Swap
`"cpu"` for `"cuda"` to run on a GPU — nothing else changes.

:::{note}
A freshly loaded NEML2 model sits on CPU with `torch.float64`
parameters. There is no separate "CUDA build" of NEML2 — the same
wheel runs on whichever devices your PyTorch install supports; you
opt in at runtime with `.to(...)`.
:::

## The input file

```{literalinclude} input.i
:language: ini
:caption: input.i
```

## Loading and inspecting placement

Load the model and check where its parameters live:

```{code-cell} ipython3
import torch
import neml2

model = neml2.load_model("input.i", "elasticity")

for name, p in model.named_parameters():
    print(f"{name:>3}: device={p.device}, dtype={p.dtype}")
```

## Moving the model and its inputs

Two pieces have to land on the target device before you call
`model(x)`:

1. The **model** — `model.to(device=...)` moves its parameters and
   buffers (recursively, for composed models).
2. The **inputs** — types in `neml2.types` (like `SR2`) accept a
   `device=` keyword in their constructors.

The cell below targets CPU. To run on a GPU, swap in
`torch.device("cuda")`:

```{code-cell} ipython3
from neml2.types import SR2

target = torch.device("cpu")  # swap for torch.device("cuda") on a CUDA box

# 1. Move model parameters/buffers.
model.to(device=target)
for name, p in model.named_parameters():
    print(f"{name:>3}: device={p.device}")

# 2. Allocate the input on the same device.
strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, device=target)
print(f"strain.device = {strain.device}")
```

## Forward pass and bringing the result home

With the model and input on the same device, the call looks just like
the CPU case. The result lives on that same device, so pull it back
with `.to(device="cpu")` if you need it for NumPy or Matplotlib:

```{code-cell} ipython3
stress = model(strain)
print(f"stress.device = {stress.device}")
stress_host = stress.to(device=torch.device("cpu"))
print(f"stress (host copy): {stress_host}")
```

If `target` had been `torch.device("cuda")`, `stress.device` would
read `cuda:0` and the `.to(device="cpu")` call would copy the result
across the host-device boundary.

## Detecting CUDA at runtime

Production code that wants to opportunistically use CUDA usually
guards on `torch.cuda.is_available()`:

```python
target = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device=target)
```

`is_available()` returns `True` when a CUDA runtime and a visible GPU
are both present. If you want to fall back gracefully when the first
CUDA call fails, wrap the first forward in a `try / except` and
reload onto CPU in the handler.

## Mixed-device errors

If the model is on one device and the input is on another, PyTorch
raises at the first op that touches both. The fix is the same in
both directions: move both ends to the same device.

On a CUDA-equipped machine, this would raise:

```python
model = neml2.load_model("input.i", "elasticity")   # CPU
strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, device="cuda")
model(strain)                                        # RuntimeError
```

## Host-device transfer cost

`.to(device=...)` is not free — it copies data between host and
device memory. A few rules of thumb:

- **Move the model once, up front.** The parameters don't change
  between calls, so copying them every time wastes bandwidth.
- **Build inputs on the device.** Pass `device=` to the constructor
  (e.g. `SR2.fill(..., device=target)`) instead of building on CPU
  and copying.
- **Pull only what you need back to CPU.** Keep the integration loop
  on GPU and only `.cpu()` the final history slice for plotting.

## Where to go next

- The same model also runs on batched inputs — see
  [](tutorials-models-vectorization).
- To read or mutate the model's parameters from Python, see
  [](tutorials-models-parameters).
