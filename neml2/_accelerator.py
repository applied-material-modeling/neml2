# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Accelerator (device-family) abstraction shared by the AOTI CLI, the Python
shim, and the C++ folder-name helper.

The on-disk artifact layout is ``<root>/<device>/<dtype>/<segment>.pt2``. The
``<device>`` segment is the lowercase ``torch.device(...).type`` string, which
must agree byte-for-byte with what the C++ loader (``device_folder_name`` in
``neml2/csrc/aoti/DeviceLayout.h``) produces. This module is the single Python
source of truth for that spelling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import torch

# Every device family NEML2 knows how to route through the compile / dispatch
# pipeline. Adding a new family (e.g. ``"mtia"``) is a one-line append plus one
# entry in ``TOOLCHAIN_CHECKS`` and, on the C++ side, one enum in
# ``neml2/csrc/aoti/DeviceLayout.h::is_accelerator``.
KNOWN_FAMILIES: tuple[str, ...] = ("cpu", "cuda", "xpu", "hip", "mps")


def parse_device_spec(s: str) -> torch.device:
    """Parse a device spec (``"cpu"``, ``"cuda"``, ``"xpu:0"``, ...) into
    a :class:`torch.device`, validating the family is one NEML2 supports.

    Cross-compilation is intentional: this accepts ``"xpu"`` on a machine
    with no XPU runtime, because the compile CLI may be producing an
    artifact for a different host. The runtime toolchain preflight
    (:func:`check_toolchain`) is the actual gate on "can this machine
    compile the requested family here."
    """
    try:
        dev = torch.device(s)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"neml2: unknown device family in spec '{s}'. "
            f"Known families: {', '.join(KNOWN_FAMILIES)}."
        ) from exc
    if dev.type not in KNOWN_FAMILIES:
        raise ValueError(
            f"neml2: unknown device family '{dev.type}' in spec '{s}'. "
            f"Known families: {', '.join(KNOWN_FAMILIES)}."
        )
    return dev


def folder_name(dev: torch.device | str) -> str:
    """Return the lowercase family name used in the ``<device>/<dtype>/``
    artifact leaf. Matches ``torch.device.type`` verbatim; kept as a named
    helper so a future rename lands in one place.
    """
    if isinstance(dev, str):
        dev = parse_device_spec(dev)
    return dev.type


def target_accelerator_family(tensors: Iterable[torch.Tensor]) -> str | None:
    """Return the first non-``cpu`` ``device.type`` seen across *tensors*,
    or ``None`` if every tensor is on CPU.

    The AOTI compile preflight uses this to decide which per-family
    toolchain check to run (:func:`check_toolchain`).
    """
    for t in tensors:
        fam = t.device.type
        if fam != "cpu":
            return fam
    return None


# ---- Per-family toolchain preflights ---------------------------------------
#
# Each entry raises ``RuntimeError`` (with a recipe) when the machine is not
# set up to compile artifacts for that family. ``check_toolchain(family)`` is
# the single dispatch point.
#
# Only CUDA has been validated end-to-end. XPU / HIP checks start as minimal
# runtime-presence probes; if a user hits a cryptic "compiler not found"
# during Inductor codegen, tighten these based on the actual error.


def _check_cpu_toolchain() -> None:
    """CPU AOTI uses whatever host C++ compiler torch itself was built to
    call. No preflight required."""
    return


def _check_cuda_toolchain() -> None:
    """Raise a recipe-bearing ``RuntimeError`` when ``nvcc`` is missing.

    Inductor's CUDA backend needs ``nvcc`` to compile the generated
    kernels. The default ``torch`` wheel bundles the CUDA *runtime* but
    not the compiler -- without ``nvcc`` you hit a cryptic
    ``OSError: CUDA_HOME environment variable is not set`` deep inside
    ``torch._inductor`` partway through the AOT compile. Surface the
    install recipe early instead.
    """
    import shutil

    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = shutil.which("nvcc")
    if not nvcc and CUDA_HOME:
        candidate = Path(CUDA_HOME) / "bin" / "nvcc"
        if candidate.exists():
            nvcc = str(candidate)
    if nvcc:
        return
    raise RuntimeError(
        "compile_model: CUDA AOTI export requires `nvcc` but none was found "
        "(neither on PATH nor under CUDA_HOME).\n"
        "\n"
        "The default torch wheel bundles the CUDA *runtime* (libcudart etc.) "
        "but not the compiler. The lightest-weight fix:\n"
        "\n"
        "    pip install nvidia-cuda-nvcc\n"
        "    export CUDA_HOME=\"$(python -c '\\\n"
        "        import pathlib, site\\n"
        "        print(next(p for sp in site.getsitepackages()\\n"
        '                   for p in pathlib.Path(sp, "nvidia").glob("cu*")\\n'
        '                   if (p / "bin/nvcc").exists()))\')"\n'
        '    export PATH="$CUDA_HOME/bin:$PATH"\n'
        "\n"
        "Or use a system install (`apt install nvidia-cuda-toolkit`, conda's "
        "`cudatoolkit-dev`, NVIDIA's network installer) and point CUDA_HOME at "
        "its root. See doc/content/installation/deps.md for details. "
        "CPU AOTI export needs none of this -- only inputs on a CUDA device "
        "trigger this check."
    )


def _check_xpu_toolchain() -> None:
    """Preflight for Intel XPU AOTI export.

    Minimal today: verify torch was built with XPU support and reports an
    available device. Intel's Inductor backend may need a DPC++/oneAPI
    compiler on PATH analogous to ``nvcc`` for CUDA -- if a user hits a
    cryptic compiler error at AOT time, tighten this check with the
    specific binary the backend shells out to.
    """
    # TODO(xpu-toolchain): once validated on Intel GPU hardware, add a
    # dpcpp/icx-equivalent PATH check with an install recipe.
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        raise RuntimeError(
            "compile_model: XPU AOTI export requires a torch build with XPU "
            "support and an available Intel GPU. The default torch wheel does "
            "not include XPU; install the XPU-enabled build:\n"
            "\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/xpu\n"
            "\n"
            "Verify with `python -c 'import torch; print(torch.xpu.is_available())'`."
        )


def _check_hip_toolchain() -> None:
    """Preflight for AMD ROCm/HIP AOTI export via native ``torch.hip``.

    Note: ROCm-built PyTorch historically aliases ``torch.cuda`` to HIP, so
    ROCm users typically spell devices as ``"cuda"`` and take the CUDA path.
    This check is for the newer, distinct ``at::kHIP`` / ``torch.hip``
    surface.
    """
    # TODO(hip-toolchain): once validated on ROCm hardware, tighten with the
    # actual codegen-compiler check (hipcc/clang) and an install recipe.
    hip = getattr(torch, "hip", None)
    if hip is None or not hip.is_available():
        raise RuntimeError(
            "compile_model: HIP AOTI export requires a torch build with native "
            "HIP support (`torch.hip`) and an available AMD GPU. Note that "
            "ROCm-built PyTorch typically exposes AMD GPUs via `torch.cuda` "
            "instead -- if that is your setup, use `--device cuda` and the "
            "existing CUDA path."
        )


def _check_mps_toolchain() -> None:
    """Preflight for Apple Silicon MPS AOTI export.

    MPS has no separate codegen-compiler analogous to nvcc, so runtime
    availability is the whole check.
    """
    if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        raise RuntimeError(
            "compile_model: MPS AOTI export requires a torch build with MPS "
            "support and an available Apple Silicon GPU. Verify with "
            "`python -c 'import torch; print(torch.backends.mps.is_available())'`."
        )


TOOLCHAIN_CHECKS: dict[str, Callable[[], None]] = {
    "cpu": _check_cpu_toolchain,
    "cuda": _check_cuda_toolchain,
    "xpu": _check_xpu_toolchain,
    "hip": _check_hip_toolchain,
    "mps": _check_mps_toolchain,
}


def check_toolchain(family: str) -> None:
    """Raise ``RuntimeError`` if this host cannot compile AOTI artifacts for
    *family*. ``family`` must be one of :data:`KNOWN_FAMILIES`.
    """
    if family not in TOOLCHAIN_CHECKS:
        raise ValueError(
            f"neml2: no toolchain preflight registered for device family '{family}'. "
            f"Known families: {', '.join(KNOWN_FAMILIES)}."
        )
    TOOLCHAIN_CHECKS[family]()
