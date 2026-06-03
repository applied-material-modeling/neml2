(install)=
# Installation

NEML2 ships as a Python wheel on
[PyPI](https://pypi.org/project/neml2/), and that is the only supported
install path for end users:

```shell
pip install neml2
```

[PyTorch](https://pytorch.org/get-started/locally/) is the sole runtime
dependency and is pulled in automatically. See [](torch-compat) for the
torch versions every wheel is regression-tested against.

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
│   └── libneml2_aoti.so     # the C++ runtime library
├── include/
│   └── neml2/csrc/          # C++ public headers
├── share/
│   ├── cmake/neml2/         # CMake config (neml2Config.cmake)
│   └── pkgconfig/           # pkg-config files (neml2.pc, …)
└── version, hash            # build metadata
```

Python consumers stop reading here: `import neml2` works as soon as the
wheel is installed.

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
ROCm build, a CPU-only build, the nightly channel — install that torch
first using the selector at
[pytorch.org/get-started](https://pytorch.org/get-started/locally/),
then run `pip install neml2`. NEML2 picks up whichever torch is already
on the import path; if it supports your accelerator, NEML2 does too. To
run a model on the GPU, pass `device="cuda"` (or any `torch.device`)
at construction or evaluation time.
