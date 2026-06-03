(dependency-management)=
# Dependencies

## Runtime

A PyPI wheel install pulls in three Python packages:

<!-- dependencies: torch.version -->
- [PyTorch](https://pytorch.org/get-started/locally/) ~= 2.12.0 — the
  tensor / autograd backend.
<!-- dependencies: pyzag.version -->
- [pyzag](https://github.com/applied-material-modeling/pyzag) — adapter
  that lets NEML2 models participate in PyTorch training loops.
<!-- dependencies: nmhit.version_min -->
- [nmhit](https://github.com/applied-material-modeling/neml2-hit) — HIT
  input-file parser.

Pip handles all three automatically. Every other library NEML2 needs at
runtime is bundled inside the wheel:

- [nlohmann/json](https://github.com/nlohmann/json) — JSON support
  (header-only; bundled under `<wheel>/include/nlohmann/`).
- libtorch — bundled via the `torch` Python package's own `lib/`
  directory; NEML2's `libneml2_aoti.so` is built with an
  `$ORIGIN/../../torch/lib` rpath that finds it automatically.

ABI compatibility between the bundled libraries and your installed
torch is verified for the version range listed in [](torch-compat).

## Source-build dependencies

Building NEML2 from source (rare; see [](build-customization)) adds:

- A C++17 compiler.
<!-- dependencies: cmake.version_min -->
- [CMake](https://cmake.org/download/) >= 3.26.1
<!-- dependencies: python.version_min -->
- [Python](https://www.python.org/downloads/) >= 3.10
- [PyTorch](https://pytorch.org/get-started/locally/) importable in the
  same Python environment used to drive the build (CMake discovers
  libtorch via the installed `torch` package by default).
- Optionally, [MPI](https://www.mpi-forum.org/) when configuring with
  `-DNEML2_MPI=ON` to enable distributed dispatching.

Everything else (nlohmann_json, etc.) is vendored as a git submodule
under `contrib/` and pulled in automatically.

## GPU acceleration

Accelerator support comes from torch, not from NEML2 directly. The torch
that `pip install neml2` pulls in by default is whatever your platform's
default PyPI index ships — already CUDA-enabled on Linux. If you need a
different variant (specific CUDA runtime, ROCm, CPU-only, nightly, …),
install that torch first using the selector at
[pytorch.org/get-started](https://pytorch.org/get-started/locally/),
then run `pip install neml2`. NEML2 transparently runs models on
whichever device the input tensors live on.
