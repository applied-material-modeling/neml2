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

"""
Export pipeline: ``torch.export`` → ``torch._inductor.aoti_compile_and_package``.

This module wraps the leading-underscore ``torch._inductor`` APIs behind a
single ``compile_model`` entry point so a torch version bump touches one file.
The resulting ``.pt2`` package is loadable from C++ via
``torch::inductor::AOTIModelPackageLoader`` (wrapped by ``neml2::aoti::Model``)
and from Python via ``load_package``.

Pinned to torch 2.12.0 — see RISK.md R-001 and decision D-001.
"""

from __future__ import annotations

import io
import struct
import warnings
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch.export import Dim, export


def _clear_elf_execstack(data: bytes) -> bytes:
    """Clear the PF_X (executable) bit from the GNU_STACK segment of an ELF SO.

    PyTorch 2.12 generates assembly constants files without a
    ``.note.GNU-stack`` annotation, causing the linker to conservatively mark
    the output ``.so`` as requiring an executable stack.  Kernels with strict
    W^X policies (``noexecstack`` enforced) refuse to ``dlopen`` such objects.
    This function strips that requirement in-place on the ELF binary so the SO
    can be loaded without kernel restrictions.
    """
    buf = bytearray(data)
    if buf[:4] != b"\x7fELF":
        return data
    endian = "<" if buf[5] == 1 else ">"
    is64 = buf[4] == 2
    if is64:
        (e_phoff,) = struct.unpack_from(endian + "Q", buf, 32)
        e_phentsize, e_phnum = struct.unpack_from(endian + "HH", buf, 54)
        flags_offset_in_phdr = 4
    else:
        (e_phoff,) = struct.unpack_from(endian + "I", buf, 28)
        e_phentsize, e_phnum = struct.unpack_from(endian + "HH", buf, 42)
        flags_offset_in_phdr = 24
    PT_GNU_STACK, PF_X = 0x6474E551, 0x1
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        (p_type,) = struct.unpack_from(endian + "I", buf, off)
        if p_type == PT_GNU_STACK:
            foff = off + flags_offset_in_phdr
            (flags,) = struct.unpack_from(endian + "I", buf, foff)
            if flags & PF_X:
                struct.pack_into(endian + "I", buf, foff, flags & ~PF_X)
    return bytes(buf)


def _patch_pt2_noexecstack(path: Path) -> None:
    """Rewrite all ELF ``.so`` files inside a ``.pt2`` zip with non-executable stack."""
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zin:
        infos = zin.infolist()
        contents = {info.filename: zin.read(info.filename) for info in infos}
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zout:
        for info in infos:
            data = contents[info.filename]
            if info.filename.endswith(".so"):
                data = _clear_elf_execstack(data)
            zout.writestr(info, data)
    path.write_bytes(out.getvalue())


def compile_model(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    package_path: str | Path,
    *,
    dynamic_batch_dim: int | None = 0,
    batch_max: int = 1 << 20,
    strict: bool = False,
    inductor_configs: dict[str, Any] | None = None,
) -> Path:
    """Export ``model`` and AOT-compile to a ``.pt2`` package at ``package_path``.

    Parameters
    ----------
    model
        A ``torch.nn.Module`` whose forward takes/returns ``torch.Tensor``s
        (or tuples thereof). Buffers are baked in as constants; ``nn.Parameter``s
        are not supported by AOTInductor's forward-only pipeline (see
        RISK.md R-007).
    example_inputs
        Positional tensors with representative dtype/device and a representative
        batch shape (typically batch=2 to force a true dynamic dim).
    package_path
        Destination path for the ``.pt2`` artifact. Created if absent;
        overwritten if present.
    dynamic_batch_dim
        Which leading axis of every input is dynamic (typically 0). If ``None``,
        all input shapes are pinned and the export specializes.
    batch_max
        Upper bound for the dynamic batch dim. PyTorch ``Dim`` requires explicit
        ``min``/``max``; an unbounded ``max`` falls back to constant shape on
        some kernels. See RISK.md R-004.
    strict
        Passed to :func:`torch.export.export`. The default remains ``False`` for
        some models that rely on the more permissive tracer. Dense
        equation-system wrappers can use ``strict=True`` to avoid non-strict
        fake-tensor constant lifting around internally seeded tangent blocks.
    inductor_configs
        Optional dict forwarded to ``aoti_compile_and_package`` as
        ``inductor_configs=...``. Use this to enable knobs like
        ``{"triton.cudagraphs": True}`` for CUDA-graph capture on graphs whose
        per-call kernel-launch overhead matters (e.g. ``DenseNewtonStep``).
        ``None`` (the default) leaves Inductor at its compile-time defaults.

    Returns
    -------
    Path
        The absolute path to the produced ``.pt2`` artifact.
    """
    import dataclasses

    # Mirror the model's forward signature in the dynamic_shapes structure.
    # `torch.export` collapses a ``*args`` (VAR_POSITIONAL) pack into a single
    # nested tuple "argument", while each named positional parameter stays a
    # top-level entry. Forward signatures come in three shapes:
    #   * pure ``*inputs``                     → every example feeds the pack
    #   * named positionals + ``*nl_params``   → first N examples are named,
    #                                            the remainder feed the pack
    #   * named positionals only               → flat 1:1 with the examples
    # so we split the example inputs at the number of named positional params.
    import inspect

    from torch._inductor import aoti_compile_and_package

    _fwd_params = list(inspect.signature(model.forward).parameters.values())
    _has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in _fwd_params)
    _n_named_positional = sum(
        p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in _fwd_params
    )

    if dynamic_batch_dim is None:
        dynamic_shapes: Any = None
    else:
        batch = Dim("batch", min=1, max=batch_max)
        leaf_spec = {dynamic_batch_dim: batch}

        def _dyn_spec(arg):
            # Mirror the input pytree structure. `torch.export` expects every
            # registered dataclass (typed tensor wrappers like SR2 / Scalar)
            # to surface as a list of its non-dropped fields — pytree
            # registration excludes ``sub_batch_ndim`` and any other static
            # metadata via ``drop_field_names``, so we skip non-pytree
            # primitive fields here too.
            if dataclasses.is_dataclass(arg) and not isinstance(arg, type):
                fields = []
                for f in dataclasses.fields(arg):
                    val = getattr(arg, f.name)
                    if isinstance(val, (torch.Tensor, tuple, list)) or (
                        dataclasses.is_dataclass(val) and not isinstance(val, type)
                    ):
                        fields.append(_dyn_spec(val))
                return fields
            if isinstance(arg, torch.Tensor):
                # Skip dynamic-batch marking for tensors with no axis to mark
                # (0-d scalars). This covers promoted-parameter inputs whose
                # shape was scalar in the source model -- they're size-pinned
                # at compile time and never get a batch axis.
                if arg.dim() <= dynamic_batch_dim:
                    return {}
                return leaf_spec
            if isinstance(arg, (tuple, list)):
                return type(arg)(_dyn_spec(x) for x in arg)
            raise TypeError(
                f"compile_model: unsupported input type {type(arg).__name__}; "
                "expected torch.Tensor, registered dataclass, or tuple/list thereof"
            )

        specs = [_dyn_spec(a) for a in example_inputs]
        if _has_varargs:
            # Named positionals stay top-level; the rest collapse into the
            # single VAR_POSITIONAL tuple entry. `torch.export` drops a
            # zero-length ``*args`` from the call structure entirely, so only
            # emit the pack entry when some inputs actually land in it.
            named = tuple(specs[:_n_named_positional])
            pack = tuple(specs[_n_named_positional:])
            dynamic_shapes = (*named, pack) if pack else named
        else:
            dynamic_shapes = tuple(specs)

    ep = export(model, example_inputs, dynamic_shapes=dynamic_shapes, strict=strict)

    package_path = Path(package_path).resolve()
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"`isinstance\(treespec, LeafSpec\)` is deprecated, use "
                r"`isinstance\(treespec, TreeSpec\) and treespec\.is_leaf\(\)` instead\."
            ),
            category=FutureWarning,
        )
        aoti_kwargs: dict[str, Any] = {"package_path": str(package_path)}
        if inductor_configs:
            aoti_kwargs["inductor_configs"] = inductor_configs
        aoti_compile_and_package(ep, **aoti_kwargs)
    # Clear GNU_STACK executable bit from compiled SOs (PyTorch 2.12 assembles
    # constants without .note.GNU-stack, causing linker to mark SO as RWE).
    _patch_pt2_noexecstack(package_path)
    return package_path


def load_package(package_path: str | Path) -> Any:
    """Load a ``.pt2`` package produced by :func:`compile_model` for in-Python use.

    This is the round-trip companion to :func:`compile_model`; the C++ side
    loads the same artifact via ``neml2::aoti::Model`` (``include/neml2/aoti/Model.h``).
    """
    from torch._inductor import aoti_load_package

    return aoti_load_package(str(Path(package_path).resolve()))
