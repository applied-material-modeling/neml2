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

"""Hardware + environment metadata collection for benchmark sweeps.

A single :func:`collect_environment` entry point returns a nested dict
containing everything we want recorded alongside the sweep results so a
future user with the same machine + same NEML2 commit can reproduce the
numbers exactly.

The dict is intentionally JSON-serializable (only str / int / float / bool /
None / list / dict). No NEML2 imports -- this module also runs under v2's
Python env during the one-time v2 baseline sweep.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def collect_environment(device: str) -> dict[str, Any]:
    """Return a serialisable dict of host + environment metadata.

    Sections present always: ``python``, ``platform``, ``torch``, ``env_vars``,
    ``cpu``, ``memory``. ``cuda`` is populated only when *device* is
    ``"cuda"`` (and at least one device is available).
    """
    out: dict[str, Any] = {
        "python": _python_info(),
        "platform": _platform_info(),
        "torch": _torch_info(),
        "env_vars": _env_var_snapshot(),
        "cpu": _cpu_info(),
        "memory": _memory_info(),
    }
    if device == "cuda":
        out["cuda"] = _cuda_info()
    return out


def folder_timestamp() -> str:
    """Compact UTC timestamp suitable for a filesystem path component."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Section collectors
# ---------------------------------------------------------------------------


def _python_info() -> dict[str, Any]:
    # ``sys.executable`` deliberately omitted: it includes the conda env
    # path (``/home/<user>/.conda/envs/...``) which leaks the runner's
    # username. Version + implementation is enough to reproduce.
    return {
        "version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
    }


def _platform_info() -> dict[str, Any]:
    # ``hostname`` deliberately omitted: HPC nodes often reveal the
    # institution (e.g. ``mom-04.egs.anl.gov``). Container indicator stays;
    # it just says docker/k8s/none and is useful for reproducibility.
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "libc": "-".join(filter(None, platform.libc_ver())) or None,
        "container_indicator": _container_indicator(),
    }


def _container_indicator() -> str | None:
    """Detect Docker / Kubernetes / generic-container hint from cgroup."""
    try:
        text = Path("/proc/1/cgroup").read_text(errors="replace")
    except OSError:
        return None
    for marker in ("docker", "kubepods", "containerd", "podman", "lxc"):
        if marker in text:
            return marker
    return None


def _torch_info() -> dict[str, Any]:
    backends_cudnn = getattr(torch.backends, "cudnn", None)
    backends_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    preferred_linalg = None
    if hasattr(torch.backends.cuda, "preferred_linalg_library"):
        try:
            preferred_linalg = str(torch.backends.cuda.preferred_linalg_library())
        except Exception:  # noqa: BLE001
            preferred_linalg = None
    return {
        "version": torch.__version__,
        "version_cuda": torch.version.cuda,
        "version_cudnn": (
            backends_cudnn.version() if (backends_cudnn and torch.cuda.is_available()) else None
        ),
        "git_version": getattr(torch.version, "git_version", None),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "allow_tf32_matmul": getattr(backends_matmul, "allow_tf32", None),
        "allow_tf32_cudnn": getattr(backends_cudnn, "allow_tf32", None),
        "preferred_linalg_library": preferred_linalg,
    }


_PERF_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "TORCHINDUCTOR_CACHE_DIR",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
)


def _env_var_snapshot() -> dict[str, str | None]:
    # For path-like env vars, only record whether they're set -- the actual
    # path almost always contains a username and is privacy-sensitive.
    PATH_LIKE = {"TORCHINDUCTOR_CACHE_DIR"}
    out: dict[str, str | None] = {}
    for name in _PERF_ENV_VARS:
        v = os.environ.get(name)
        if v is None:
            out[name] = None
        elif name in PATH_LIKE:
            out[name] = "<set>"
        else:
            out[name] = v
    return out


def _cpu_info() -> dict[str, Any]:
    model, vendor = None, None
    physical_cores = None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(errors="replace")
        for line in cpuinfo.splitlines():
            if model is None and line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
            elif vendor is None and line.startswith("vendor_id"):
                vendor = line.split(":", 1)[1].strip()
            elif physical_cores is None and line.startswith("cpu cores"):
                try:
                    physical_cores = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            if model and vendor and physical_cores is not None:
                break
    except OSError:
        pass

    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            affinity = None

    return {
        "model": model,
        "vendor": vendor,
        "logical_cores": os.cpu_count(),
        "physical_cores": physical_cores,
        "frequency_mhz_min": _read_cpu_freq("cpuinfo_min_freq"),
        "frequency_mhz_max": _read_cpu_freq("cpuinfo_max_freq"),
        "frequency_mhz_base": _read_cpu_freq("base_frequency"),
        "governor": _read_cpu_attr("scaling_governor"),
        "numa_nodes": _count_numa_nodes(),
        "affinity_mask": affinity,
    }


def _read_cpu_attr(name: str) -> str | None:
    try:
        return Path(f"/sys/devices/system/cpu/cpu0/cpufreq/{name}").read_text().strip()
    except OSError:
        return None


def _read_cpu_freq(name: str) -> int | None:
    raw = _read_cpu_attr(name)
    if raw is None:
        return None
    try:
        return int(raw) // 1000  # cpufreq sysfs reports kHz; convert to MHz
    except ValueError:
        return None


def _count_numa_nodes() -> int | None:
    try:
        return sum(
            1 for p in Path("/sys/devices/system/node/").iterdir() if p.name.startswith("node")
        )
    except OSError:
        return None


def _memory_info() -> dict[str, Any]:
    try:
        text = Path("/proc/meminfo").read_text(errors="replace")
    except OSError:
        return {"total_gb": None, "available_gb": None}
    total_kb, available_kb = None, None
    for line in text.splitlines():
        if total_kb is None and line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif available_kb is None and line.startswith("MemAvailable:"):
            available_kb = int(line.split()[1])
        if total_kb is not None and available_kb is not None:
            break
    return {
        "total_gb": round(total_kb / 1024**2, 1) if total_kb else None,
        "available_gb": round(available_kb / 1024**2, 1) if available_kb else None,
    }


def _cuda_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    cap_major, cap_minor = torch.cuda.get_device_capability(idx)
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "device_index": idx,
        "device_name": props.name,
        "compute_capability": f"{cap_major}.{cap_minor}",
        "total_memory_gb": round(props.total_memory / 1024**3, 1),
        "multi_processor_count": props.multi_processor_count,
        "driver_version": _nvidia_smi_query("driver_version"),
        "cuda_runtime_version": _cuda_runtime_version(),
        "nvcc_version": _nvcc_version(),
        "nvidia_smi_snapshot": _nvidia_smi_q(),
    }


def _cuda_runtime_version() -> str | None:
    # torch.version.cuda is the version torch was BUILT against; the runtime
    # version is what's actually loaded. They're usually the same on a clean
    # install but can drift in messier environments. Report both -- the
    # build version sits in `torch.version_cuda`.
    if not torch.cuda.is_available():
        return None
    # torch.cuda._driver_version() and torch.cuda._runtime_version() exist on
    # recent torch versions; both are private. Fall back to nvidia-smi /
    # torch.version.cuda metadata when they're not exposed.
    runtime_getter = getattr(torch.cuda, "_runtime_version", None)
    if runtime_getter is not None:
        try:
            v = runtime_getter()
            major, minor, patch = v // 1000, (v % 1000) // 10, v % 10
            return f"{major}.{minor}.{patch}"
        except Exception:  # noqa: BLE001
            return None
    return None


def _nvidia_smi_query(field: str) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return out.strip().splitlines()[0].strip() or None
    except (subprocess.SubprocessError, OSError, IndexError):
        return None


def _nvidia_smi_q() -> dict[str, str | None] | None:
    """Compact snapshot of GPU state via ``nvidia-smi`` query flags.

    Captures the perf-relevant fields without the noise of the full
    ``-q`` XML dump. Returns None if nvidia-smi is unavailable.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    fields = (
        "compute_mode",
        "persistence_mode",
        "power.max_limit",
        "power.draw",
        "clocks.gr",
        "clocks.mem",
        "clocks.max.gr",
        "clocks.max.mem",
        "temperature.gpu",
        "memory.used",
        "memory.free",
        "ecc.mode.current",
    )
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    values = [v.strip() for v in line.split(",")]
    if len(values) != len(fields):
        return None
    return dict(zip(fields, values, strict=True))


def _nvcc_version() -> str | None:
    if shutil.which("nvcc") is None:
        return None
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return None
    m = re.search(r"release\s+(\S+),\s+V([\S]+)", out)
    if m:
        return m.group(2)
    return out.strip().splitlines()[-1].strip() if out.strip() else None


__all__ = ["collect_environment", "folder_timestamp"]
