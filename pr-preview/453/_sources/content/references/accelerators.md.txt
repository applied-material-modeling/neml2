(accelerators-page)=
# Accelerators (device families)

NEML2 targets any device family torch itself supports. Today that means CPU,
CUDA, XPU (Intel GPU), HIP (AMD GPU via native `torch.hip`), and MPS (Apple
Silicon). The compile CLI, the on-disk artifact layout, the Python shim, and
the C++ AOTI loader + dispatcher all speak the same lowercase family names
(`cpu`, `cuda`, `xpu`, `hip`, `mps`) — the runtime is device-opaque, so adding
a new family is a small change at a handful of choke points.

:::{note}
CPU and CUDA are the two families NEML2 exercises end-to-end in CI. XPU, HIP
(native), and MPS are **wired** and unit-tested at the string-parsing and
metadata boundaries, but not yet validated on the hardware they target. If you
run on one of those, please open an issue with the outcome — the code paths
share the same generic runtime, so parity is expected, but real-hardware
confirmation is welcome.
:::

## Accepted `--device` values

`neml2-compile --device` and `neml2-run --device` accept any of:

| Family | Meaning |
|---|---|
| `cpu` | Host CPU (always available) |
| `cuda` | NVIDIA GPU via CUDA |
| `xpu` | Intel GPU via `torch.xpu` |
| `hip` | AMD GPU via native `torch.hip` (see note below) |
| `mps` | Apple Silicon via `torch.backends.mps` |

`neml2-compile --device` takes bare family names (one artifact per family);
pass more than one to build side-by-side artifacts (`--device cpu cuda xpu`).
`neml2-run --device` additionally accepts indexed forms (`cuda:1`, `xpu:0`)
because it sets `torch.set_default_device` and a specific index is meaningful.

:::{note}
**ROCm-built PyTorch aliases `torch.cuda` to HIP.** If you installed
PyTorch from the ROCm wheel index, your AMD GPU is addressed as `"cuda"` (not
`"hip"`) and works under the existing CUDA path with no changes. `"hip"` in
the accepted list is for the newer, distinct `torch.hip` / `at::kHIP`
surface.
:::

## Per-family requirements

Each family needs the matching torch build. `neml2-compile` runs a per-family
preflight ({py:mod}`neml2._accelerator`'s `check_toolchain`) as soon as example
inputs on that device reach the compile step; if the toolchain is missing you
get a recipe rather than a cryptic Inductor error deep in the graph.

**CUDA.** Inductor's CUDA backend needs `nvcc` at compile time (the default
torch wheel ships the CUDA runtime but not the compiler). Install
`nvidia-cuda-nvcc` (pip) or the system CUDA toolkit; the error message includes
the exact recipe.

**XPU.** Install the XPU-enabled torch build:

```console
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Verify with `python -c 'import torch; print(torch.xpu.is_available())'`. NEML2's
current XPU toolchain check only confirms runtime availability; if you hit a
compiler error inside Inductor's XPU codegen, please open an issue with the
message so we can tighten the preflight.

**HIP.** Requires a torch build with native `torch.hip`. If you have ROCm
PyTorch (which routes `torch.cuda` to HIP), just use `--device cuda`.

**MPS.** Available on Apple Silicon with a stock torch build. No separate
compiler is needed.

## On-disk layout

`neml2-compile` writes one subfolder per targeted family under the artifact
root, named by the lowercase family string:

```text
<output-dir>/<name>/
  metadata.json
  cpu/float64/  *.pt2
  cuda/float64/ *.pt2
  xpu/float64/  *.pt2
```

The C++ loader (`neml2::aoti::Model`, `neml2::aoti::DispatchedModel`) picks the
folder from `at::Device(dev).type()` at load time, so the Python writer and the
C++ reader always agree on the spelling.

## Dispatcher: mixing families

`neml2::aoti::DispatchedModel` accepts any mix of families in its `WorkScheduler`
configuration — the scheduler chunks a batched call across the pool. The
per-family rules (from `MPISimpleScheduler` and `StaticHybridScheduler`):

- At most one CPU worker (concurrent CPU workers would oversubscribe the
  intra-op thread pool).
- Within each accelerator family, indices must be unique; a bare family entry
  (e.g. `cuda`) cannot mix with pinned ones (`cuda:0`, `cuda:1`) from the same
  family, because bare could alias one of the pinned devices.
- Different families are independent — `{"cpu", "cuda:0", "xpu:0"}` is legal.

See [](model-dispatch) for the full scheduler surface.

## Adding a new accelerator family

Every new torch accelerator (MTIA, PrivateUse1-backed hardware, etc.) is a
one-line extension at two places:

1. Python: append the family name to `KNOWN_FAMILIES` in
   {py:mod}`neml2._accelerator` and add a `TOOLCHAIN_CHECKS` entry (a
   `torch.<x>.is_available()` probe is enough to start).
2. C++: add the enum to `is_accelerator()` in
   `neml2/csrc/aoti/DeviceLayout.h`. `c10::DeviceTypeName(..., true)` already
   knows how to spell every torch-known device type, so the folder-name
   mapping needs no change.

No changes are needed in the loader, ops, solve, jacobian, or dispatcher code —
they route `at::Device` opaquely.
