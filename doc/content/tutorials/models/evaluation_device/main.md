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

NEML2 models are `torch.nn.Module` subclasses, so a model's *evaluation
device* is just a `torch.device`. This tutorial walks through the
mechanics of moving a model (plus its parameters and buffers) and its
inputs onto a target device, what happens when devices don't match,
and where the host-device transfer cost shows up.

The runnable cells stay on CPU so the doc build is portable; the API
they exercise (`.to(device=...)`, `device=` kwarg on constructors) is
the same one you would use with a CUDA, MPS, XPU, or any other torch
device — only the device string changes.

## What "device" means here

NEML2 inherits PyTorch's notion of a device. A device is identified
by a *type* (`"cpu"`, `"cuda"`, …) and an optional *index*
(`"cuda:0"`, `"cuda:1"`, …). CPU is the default; everything else is
an opt-in via `.to(...)`.

:::{note}
By default, every freshly loaded NEML2 model sits on CPU with
`torch.float64` parameters. There is no separate "CUDA build" of
NEML2 — the same wheel works on every device PyTorch supports;
you opt in at runtime with `.to(...)`.
:::

## The input file

```{literalinclude} input.i
:language: ini
:caption: input.i
```

## Loading and inspecting placement

A freshly loaded model owns its parameters as `torch.Tensor`s. We can
ask them where they live:

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

1. The **model** itself — `.to(device=...)` recurses through all
   parameters and buffers (and through every child `Model` in a
   composition).
2. The **inputs** to the forward operator — the typed tensor wrappers
   in `neml2.types` accept a `device=` keyword on their constructors
   and on `.to(...)`.

The cell below demonstrates the API by targeting CPU. To target CUDA,
substitute `torch.device("cuda")` (or any other supported torch
device); nothing else in the code changes:

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

Once everything is co-located the call site looks identical to the
CPU case. The output lives on whichever device the forward ran on, so
if you need the result back in NumPy / Matplotlib land you pull it
back with `.to(device="cpu")`:

```{code-cell} ipython3
stress = model(strain)
print(f"stress.device = {stress.device}")
stress_host = stress.to(device=torch.device("cpu"))
print(f"stress (host copy): {stress_host}")
```

If `target` had been `torch.device("cuda")`, `stress.device` would
read `cuda:0` and the `.to(device="cpu")` call would copy the result
across the PCIe bus.

## Detecting CUDA at runtime

Production code that wants to opportunistically use CUDA usually
guards on `torch.cuda.is_available()`:

```python
target = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device=target)
```

`is_available()` returns `True` whenever a CUDA runtime and a visible
GPU are both present, but it does not guarantee that the installed
PyTorch wheel can actually run kernels on that GPU — older compute
capabilities can be silently dropped by newer wheels. Wrap the first
forward in `try / except torch.AcceleratorError` if you need to
survive that case.

## Mixed-device errors

If the model's parameters live on one device and the input tensor on
another, PyTorch raises at the first op that needs both operands.
The error message is verbose but the root cause is always the same:
one of the two ends forgot a `.to(device)`. The fix is to move *both*
the model and the input to the same device before calling the model.

For example, on a CUDA-equipped machine the following would raise:

```python
model = neml2.load_model("input.i", "elasticity")   # CPU
strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, device="cuda")
model(strain)                                        # RuntimeError
```

## Host-device transfer cost

`.to(device=...)` is not free — it copies data over the PCIe bus (or
NVLink, etc.). Three rules of thumb:

- **Move the model once, up front.** Parameters and buffers are
  invariant across a batch sweep; copying them on every call wastes
  bandwidth.
- **Build inputs on the device, don't construct on CPU then copy.**
  The `device=` keyword on `SR2.fill`, `Scalar(...)`, `R2(...)`, etc.
  allocates directly on the target device.
- **Pull only what you need back to CPU.** A common pattern is to
  keep the entire integration loop on GPU and only `.cpu()` the
  final history slice for post-processing or plotting.

## Where to go next

- The forward operator scales naturally across batch dimensions —
  see [](tutorials-models-vectorization) for how to feed large
  batches through the same model evaluation we just ran on a single
  sample.
- Once you have model + inputs co-located on a device, the next
  questions are usually about hooking parameters up to training
  loops — [](tutorials-models-parameters) covers reading and
  mutating parameters from Python.
