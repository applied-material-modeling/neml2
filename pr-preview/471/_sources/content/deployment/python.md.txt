(python-integration)=
# Python integration

The most direct way to use neml2 is from Python: `pip install`, load a model
from its input file, and evaluate it like any `torch.nn.Module`. This page is
the *setup* side — installing neml2 in your application, the dependency
footprint, and where model artifacts come from. The actual evaluation API for
each route lives in the per-route references linked at the bottom.

## Install

```shell
pip install neml2
```

The wheel bundles the compiled runtime (`libneml2.so`) and pulls in a matching
`torch`. For a specific CPU / CUDA / ROCm / nightly torch build — or an
editable install from source — see [](installation).

## Dependency footprint

A Python app that embeds neml2 depends on **neml2 + torch** (torch is the
tensor backend and the bulk of the install size). There is no separate
libtorch download and no system NEML2 install: everything resolves through the
two wheels in your environment. Pin both for reproducible deployments —
neml2's pinned torch is recorded in its wheel metadata.

## Obtaining a model artifact

Which artifact you ship depends on the route:

- **A HIT input file (`.i`)** — the model definition, consumed by the eager
  routes (`py-eager`, `py-jit`). This is the model's source of truth.
- **A compiled AOT-Inductor package (`.pt2` + metadata + stub)** — produced by
  `neml2-compile` for the `py-aoti` route, when you want to ship a frozen,
  fast-loading model without re-deriving it at runtime. See [](aoti-packages)
  for the format and [](cli-utilities) for `neml2-compile`.

## Import and device

```python
import neml2

model = neml2.load_model("input.i", "my_model")
model.to(device="cuda", dtype=torch.float64)  # standard nn.Module placement
```

neml2 models are plain `torch.nn.Module`s, so device and dtype follow the usual
torch conventions. Python evaluation is single-device; to spread one batched
call across several devices you compile and use the C++ dispatcher
([](model-dispatch)).

## Now evaluate

With neml2 installed and an artifact in hand, pick the route that fits:

- [](py-eager) — load and call the model directly; the default for development,
  interactive use, and autograd training.
- [](py-jit) — accelerate the in-process graph with `neml2.compile`, primarily
  for pyzag training loops.
- [](py-aoti) — load and run a compiled `.pt2` package from Python.

For a hands-on first run, start with
[](tutorials-models-running-your-first-model).

## See also

- [](installation) — torch variants, CUDA/ROCm, building from source.
- [](deployment-overview) — all six routes and how to choose.
- [](external-project-integration) — the C++ counterpart to this page.
