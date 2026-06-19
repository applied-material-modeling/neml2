(install)=
# Installation

NEML2 ships as a Python wheel on
[PyPI](https://pypi.org/project/neml2/), and that's the recommended
install path for end users:

```shell
pip install neml2
```

[PyTorch](https://pytorch.org/get-started/locally/) and a couple of
small helper packages are the only runtime dependencies, and pip pulls
them in automatically. See [](torch-compat) for the torch versions
every wheel is regression-tested against, and [](dependency-management)
for the full dependency list.

Verify the install:

```python
import neml2
print(neml2.__version__)
```

## What you get

A single `pip install neml2` lays down everything needed to use NEML2
from **either** Python or C++ — no second install step, no separate
LibTorch download.

The wheel installs into your site-packages under `neml2/`:

```
<site-packages>/neml2/
├── __init__.py              # `import neml2` entry point
├── aoti/, models/, ...      # the Python package
├── lib/
│   └── libneml2.so          # the C++ runtime library
├── include/
│   └── neml2/csrc/          # C++ public headers
├── share/
│   ├── cmake/neml2/         # CMake config (neml2Config.cmake)
│   └── pkgconfig/           # pkg-config files (neml2.pc, …)
└── version, hash            # build metadata
```

Python consumers can skip ahead to [](getting-started); the next
section on torch variants is worth a look if you need a specific CUDA
/ ROCm / nightly build.

C++ consumers point their build system at the relevant subdirectories of
the wheel — both CMake and `pkg-config` are supported. The wiring is
covered in [](external-project-integration).

## Source builds

If you are contributing to NEML2 itself or need a build flavor the
wheels don't ship (CUDA-only toolchain, debug build, sanitizer build,
…), see [](build-customization) for the developer source-build
workflow.

## Choosing a torch variant

`pip install neml2` pulls in whichever torch your platform's default
PyPI index serves — that's the right choice for most users and needs
no further action.

If you need a *specific* torch variant — a particular CUDA runtime, a
ROCm build, a CPU-only build, the nightly channel — install it in two
steps:

1. Install the torch variant you want via the selector at
   [pytorch.org/get-started](https://pytorch.org/get-started/locally/).
2. Run `pip install neml2`.

NEML2 picks up whichever torch is already on the import path, so any
accelerator torch supports works too. For moving a model onto a device
and allocating inputs there, see [](tutorials-models-evaluation-device).
