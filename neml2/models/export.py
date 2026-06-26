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
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch.export import Dim, export

from .._warnings import TORCH_JIT_PY314, TORCH_TREESPEC_LEAFSPEC, ignore_warnings


def _patch_inductor_int_array_cache_key() -> None:
    """Backport pytorch/pytorch#178147 onto torch 2.12's inductor cpp_wrapper.

    ``CppWrapperCpu.codegen_int_array_var`` caches the emitted ``int_array_N``
    name keyed on ``(int_array, id(writeline), known_statically, id(graph))``.
    ``writeline`` is typically a bound method (``self.wrapper_call.writeline``);
    Python builds a fresh bound-method instance on every attribute access, so
    ``id(writeline)`` is transient and the allocator routinely recycles the
    same address for a different writeline scope. The cache then returns an
    ``int_array_N`` name without re-emitting its declaration in the new scope,
    producing wrapper.cpp lines like ``aoti_torch_cpu_bmm_out(..., int_array_0,
    ...)`` with no matching ``static const int64_t int_array_0[]`` declaration.
    The C++ build fails with ``'int_array_0' was not declared in this scope``.

    The upstream fix keys on ``id(writeline.__self__)`` instead so the cache
    survives bound-method recreation. This patch applies the same change in
    place at import time; remove it once we move past torch 2.12.

    Triggered in NEML2 by ``scpdecoup`` (two sequential ``ImplicitUpdate``
    segments produce a ``linalg_solve`` proxy-executor fallback that takes the
    cache-corrupting path); will also fire on any future model with similar
    structure.
    """
    try:
        from torch._inductor.codegen import cpp_wrapper_cpu as _ccw  # noqa: PLC0415
    except ImportError:
        return  # nothing to patch; downstream torch reshuffle
    cls = getattr(_ccw, "CppWrapperCpu", None)
    if cls is None or not hasattr(cls, "codegen_int_array_var"):
        return
    if getattr(cls, "_neml2_int_array_patched", False):
        return
    _orig = cls.codegen_int_array_var

    def codegen_int_array_var(self, int_array: str, writeline, known_statically=False, graph=None):
        writeline_key = id(writeline.__self__) if hasattr(writeline, "__self__") else id(writeline)
        cache_key = (int_array, writeline_key, known_statically, id(graph) if graph else None)
        if cache_key not in self.codegen_int_array_var_cache:
            self.codegen_int_array_var_cache[cache_key] = self._codegen_int_array_var_impl(
                int_array, writeline, known_statically
            )
        return self.codegen_int_array_var_cache[cache_key]

    codegen_int_array_var.__doc__ = _orig.__doc__
    cls.codegen_int_array_var = codegen_int_array_var
    cls._neml2_int_array_patched = True


_patch_inductor_int_array_cache_key()


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


def _wants_cuda(example_inputs: tuple[Any, ...]) -> bool:
    """True iff any tensor leaf inside *example_inputs* is on a CUDA device.

    Mirrors :func:`_walk_tensors`'s recursion so dataclass-wrapped typed
    tensors and ``(tuple|list)`` packs are inspected too.
    """
    return any(t.is_cuda for arg in example_inputs for t in _walk_tensors(arg))


def _check_cuda_toolchain_available() -> None:
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


def _walk_tensors(arg: Any):
    """Yield every ``torch.Tensor`` leaf inside a (possibly nested) export arg.

    Mirrors the recursion in :func:`compile_model._dyn_spec` so the same
    structural assumptions apply: dataclass wrappers (typed tensors) are
    descended via their non-static fields; tuples/lists are walked
    element-wise; primitive non-tensor leaves are silently skipped.
    """
    import dataclasses as _dc  # noqa: PLC0415

    if isinstance(arg, torch.Tensor):
        yield arg
        return
    if _dc.is_dataclass(arg) and not isinstance(arg, type):
        for f in _dc.fields(arg):
            val = getattr(arg, f.name)
            if isinstance(val, (torch.Tensor, tuple, list)) or (
                _dc.is_dataclass(val) and not isinstance(val, type)
            ):
                yield from _walk_tensors(val)
        return
    if isinstance(arg, (tuple, list)):
        for x in arg:
            yield from _walk_tensors(x)


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

    # Preflight: if any example input lives on a CUDA device, the Inductor
    # backend will JIT/AOT-compile CUDA kernels and that requires ``nvcc``
    # on the host -- torch's default wheel bundles the CUDA *runtime* but
    # not the compiler. Detect early and emit a recipe instead of letting
    # the user hit the cryptic ``OSError: CUDA_HOME environment variable
    # is not set`` deep inside Inductor.
    if _wants_cuda(example_inputs):
        _check_cuda_toolchain_available()

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

        # Mirror the typed-tensor pytree drop list in `neml2/types/_pytree.py`:
        # every field listed here is registered as ``drop_field_names`` on
        # the dataclass, so torch.export's flatten does not surface it as a
        # subtree. ``_dyn_spec`` must skip the same names — otherwise an
        # empty ``sub_batch_state = ()`` field is treated as a pytree
        # subtree and the resulting spec has more entries than torch
        # expects ("Node arity mismatch; expected 1, but got 3").
        _DROPPED_PYTREE_FIELDS = frozenset(
            (
                "sub_batch_ndim",
                "sub_batch_state",
                "sub_batch_meta",
                "sub_batch_labels",
                "k_ndim",
                "k_state",
                "k_pairing",
            )
        )

        def _dyn_spec(arg):
            # Mirror the input pytree structure. `torch.export` expects every
            # registered dataclass (typed tensor wrappers like SR2 / Scalar)
            # to surface as a list of its non-dropped fields — pytree
            # registration excludes ``sub_batch_ndim`` and the sub-batch
            # metadata tuples via ``drop_field_names``, so we skip those
            # here too.
            if dataclasses.is_dataclass(arg) and not isinstance(arg, type):
                fields = []
                for f in dataclasses.fields(arg):
                    if f.name in _DROPPED_PYTREE_FIELDS:
                        continue
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

        # Opt out of torch's 0/1 specialization on the dynamic batch dim.
        # ``Dim("batch", min=1, ...)`` still adds an internal ``> 1`` assertion
        # (torch ``symbolic_shapes._add_assertion``) that bakes a static
        # ``batch_max >= 2`` precondition into the wrapper's xnumel math --
        # downstream Triton kernels then issue out-of-bounds pointer arithmetic
        # when called at batch=1. Marking the dim as unbacked tells torch to
        # treat its value as data-dependent and emit a true dynamic kernel.
        #
        # ``min`` / ``max`` constraint kwargs landed in a torch 2.12 point
        # release; older 2.12 wheels (some CI runners pick those up via
        # cache or constraint ordering) reject them. Detect the signature
        # and pass them only when supported -- losing the range
        # constraints is a soft degradation: the dim stays unbacked, so
        # Triton still emits a dynamic kernel.
        import inspect as _inspect  # noqa: PLC0415

        _mark_unbacked = torch._dynamo.decorators.mark_unbacked
        _has_min_max = "min" in _inspect.signature(_mark_unbacked).parameters
        _bound_kwargs = {"min": 1, "max": batch_max} if _has_min_max else {}
        for a in example_inputs:
            for t in _walk_tensors(a):
                if t.dim() > dynamic_batch_dim:
                    _mark_unbacked(t, dynamic_batch_dim, **_bound_kwargs)

    package_path = Path(package_path).resolve()
    package_path.parent.mkdir(parents=True, exist_ok=True)
    # The export + inductor compile trip two torch-internal deprecations the caller
    # cannot act on (torch calling its own deprecated APIs): the LeafSpec pytree
    # call during ``export()``'s deepcopy, and ``torch.jit.script_method`` while
    # inductor imports ``torch.utils.mkldnn``. Silence them only for the lowering;
    # the specs live in ``neml2/_warnings.py`` (shared with the tests).
    with ignore_warnings(TORCH_TREESPEC_LEAFSPEC, TORCH_JIT_PY314):
        ep = export(model, example_inputs, dynamic_shapes=dynamic_shapes, strict=strict)
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
