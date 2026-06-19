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

"""Native-Model-compatible HIT shim around the AOTI pybind binding.

Registers ``AOTIModel`` with the native factory so that an HIT block like

    [Models]
      [my_model]
        type = AOTIModel
        artifact_path = '/abs/path/to/aoti/my_model'
      []
    []

loads through ``neml2.load_input`` exactly the same way as any other
native model -- and presents a ``Model``-compatible surface
(``input_spec`` / ``output_spec`` / positional-tuple ``__call__``) so the
native drivers (``TransientDriver``, ``TransientRegression``, ...) can drive
it without special-casing.

This is a thin alias layer. The underlying runtime is the C++
``neml2::aoti::Model`` exposed via the ``_aoti`` pybind module; this shim
only handles HIT loading + typed-wrapper marshalling at the boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from .. import types as _types
from ..factory import register_neml2_object
from ..schema import HitSchema, dependency, option
from ..types import TensorWrapper
from ._aoti import Model as _BoundModel

if TYPE_CHECKING:
    import nmhit

    from ..factory import _NativeInputFile


@register_neml2_object("AOTIModel")
class AOTIModel(nn.Module):
    """HIT-loadable wrapper around :class:`neml2.aoti.Model`.

    Constructed from a HIT ``[Models]`` block with an ``artifact_path`` option
    pointing at the per-device artifact folder produced by ``neml2-compile``
    (the folder holding one ``<device>/`` subfolder per compiled device). The
    subfolder for the current ``torch.get_default_device()`` is loaded -- so
    ``neml2-run --device cuda`` (which sets the default device) picks ``cuda/``.
    Eager and single-device: no dispatch happens here.

    Plays the native-Model role: ``input_spec`` and ``output_spec`` are
    populated from the metadata's ``var_type`` fields; ``__call__`` takes
    ``TensorWrapper`` positional args in ``input_spec`` order, unwraps them
    to raw tensors, runs the AOTI ``forward`` graph, and wraps each output
    back in its declared type. Promoted parameters (if any) live on the
    underlying binding's ``named_parameters()`` and are *not* part of
    ``input_spec`` -- the caller mutates them in place via
    ``self._inner.named_parameters()`` to drive runtime-flexible behavior.
    """

    #: Inherits from ``nn.Module`` rather than :class:`neml2.model.Model` so
    #: the bound ``torch::inductor::AOTIModelPackageLoader`` runtime drives
    #: evaluation; the explicit class attribute keeps it in the [Models]
    #: section of the syntax catalog despite not subclassing Model.
    SECTION = "Models"

    hit = HitSchema(
        option(
            "artifact_path",
            str,
            "Absolute path to the per-device artifact folder produced by "
            "``neml2-compile`` (contains one ``<device>/`` subfolder per compiled "
            "device). The subfolder matching ``torch.get_default_device()`` is loaded.",
        ),
        dependency(
            "solver",
            "get_solver",
            "Solver whose convergence / line-search settings configure the implicit "
            "Newton solve. Schema v4+ no longer bakes these into the artifact; the "
            "stub ``.i`` carries the ``[Solvers]`` block and it is forwarded to the "
            "C++ runtime at load. Defaults apply for forward-only models.",
            default=None,
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> AOTIModel:
        artifact_str = node.param_str("artifact_path")
        # Absolute per `neml2-compile`; tolerate a relative path by resolving it
        # against the input file's directory.
        artifact_path = Path(artifact_str)
        if not artifact_path.is_absolute():
            artifact_path = factory._path.parent / artifact_path
        artifact_path = artifact_path.resolve()

        # Load the subfolder for the current default device (cpu / cuda).
        device = torch.get_default_device()
        device_dir = artifact_path / device.type
        metas = sorted(device_dir.glob("*_meta.json")) if device_dir.is_dir() else []
        if not metas:
            raise FileNotFoundError(
                f"AOTIModel({node.path()!r}): no artifact compiled for device "
                f"{device.type!r} under {artifact_path} (looked in {device_dir}). "
                f"Recompile with `neml2-compile --device {device.type}` or change the "
                f"default device."
            )
        if len(metas) > 1:
            raise RuntimeError(
                f"AOTIModel({node.path()!r}): multiple '*_meta.json' files in "
                f"{device_dir}; expected exactly one compiled model."
            )

        model = cls(metas[0])
        solver_name = node.param_optional_str("solver", "")
        if solver_name:
            model._apply_solver_config(factory.get_solver(solver_name))
        return model

    def _apply_solver_config(self, solver) -> None:
        """Forward a Python solver's config to the C++ runtime.

        Reuses the solver wrapper's own ``_solver_config()`` -- the exact dict
        the eager path passes to ``newton_solve_eager`` -- so the compiled and
        eager solves are configured from a single source of truth.
        """
        self._inner.set_solver_config(**solver._solver_config())

    def __init__(self, meta_path: str | Path) -> None:
        super().__init__()
        meta_path = Path(meta_path)
        self._inner = _BoundModel(str(meta_path))
        # Pull typed-wrapper class info from the metadata. The binding's
        # `input_names`/`input_sizes` give names + flat sizes; for the
        # native-driver surface we additionally need the TensorWrapper class
        # (e.g. Scalar vs SR2) which the metadata records under `var_type`.
        with open(meta_path) as f:
            meta = json.load(f)
        self.input_spec = self._spec_from_meta(meta["inputs"])
        self.output_spec = self._spec_from_meta(meta["outputs"])
        # Per-input sub-batch shape (empty tuple when the input has none).
        # ``_broadcast_to_common_batch`` consults this to split each input's
        # batch axes into (dyn, sub) and broadcast only the dyn portion --
        # the sub-batch axis is part of the input's identity and must not
        # be flattened against a global (no-sub-batch) sibling.
        self.input_sub_batch: dict[str, tuple[int, ...]] = {
            info["name"]: tuple(info.get("sub_batch_shape", ())) for info in meta["inputs"]
        }
        # Per-input / per-output sub-batch labels (empty tuple when the
        # variable carries none). Persisted at export time via
        # :func:`~neml2.cli.aoti_export._var_infos`; re-attached when
        # wrapping AOTI raw outputs back into typed wrappers so the
        # per-axis label dispatch (preserved-label storage, BLOCK-aware
        # matmul) survives the export-and-load round-trip.
        self.input_labels: dict[str, tuple[str, ...]] = {
            info["name"]: tuple(info.get("sub_batch_labels", ())) for info in meta["inputs"]
        }
        self.output_labels: dict[str, tuple[str, ...]] = {
            info["name"]: tuple(info.get("sub_batch_labels", ())) for info in meta["outputs"]
        }

    @staticmethod
    def _spec_from_meta(infos: list[dict]) -> dict[str, type[TensorWrapper]]:
        """Map each metadata ``var_type`` string to the TensorWrapper class."""
        spec: dict[str, type[TensorWrapper]] = {}
        for info in infos:
            name = info["name"]
            type_name = info["var_type"]
            type_cls = getattr(_types, type_name, None)
            if type_cls is None or not (
                isinstance(type_cls, type) and issubclass(type_cls, TensorWrapper)
            ):
                raise TypeError(
                    f"AOTIModel: metadata reports var_type={type_name!r} for "
                    f"variable {name!r}, but neml2.types has no such "
                    f"TensorWrapper subclass."
                )
            spec[name] = type_cls
        return spec

    def _check_tensor(self, t: torch.Tensor, name: str, *, kind: str) -> None:
        """Strict device + dtype check; raise TypeError on mismatch.

        ``kind`` is ``'input'`` or ``'parameter'`` and shows up in the error
        message so the caller can tell whether to fix the call site or the
        parameter setter. Only floating-point dtype is checked -- bool /
        integer tensors are passed through unchanged because their widths
        are not part of the AOTI compile-pin.

        The device comparison is index-aware. ``torch.device('cuda')`` and
        ``torch.device('cuda:0')`` refer to the same physical GPU but
        ``__eq__`` reports them unequal; we treat them as equal whenever
        the target lacks an explicit index. Same-type + same-index (when
        the target pins an index) is required.
        """
        target_device = self._inner.device
        target_dtype = self._inner.dtype
        device_ok = t.device.type == target_device.type and (
            target_device.index is None or t.device.index == target_device.index
        )
        problems = []
        if not device_ok:
            problems.append(f"device={t.device} (expected {target_device})")
        if t.is_floating_point() and t.dtype != target_dtype:
            problems.append(f"dtype={t.dtype} (expected {target_dtype})")
        if not problems:
            return
        raise TypeError(
            f"AOTIModel {kind} {name!r}: {', '.join(problems)}. "
            f"The .pt2 artifact is compile-pinned to "
            f"(device={target_device}, dtype={target_dtype}) and the runtime "
            f"refuses to silently coerce -- silent coercion would mask the "
            f"silent-garbage / SEGV failure mode where a stride-based kernel "
            f"reads off the end of a mistyped buffer. Cast the tensor "
            f"explicitly with ``.to(device=..., dtype=...)`` at the call "
            f"site, or set process-wide defaults (e.g. "
            f"``torch.set_default_dtype(torch.float64)`` + "
            f"``torch.set_default_device(...)``) before constructing inputs."
        )

    def forward(self, *args: TensorWrapper, v=None, v2=None, vh=None):
        """Drive the AOTI graph.

        Accepts ``input_spec`` positional args as ``TensorWrapper`` instances
        (mirroring the native ``ComposedModel`` boundary). Returns a tuple of
        typed wrappers in ``output_spec`` order -- always a tuple, even for
        single-output models, so consumers can iterate uniformly.

        ``v`` / ``v2`` / ``vh`` (the native chain-rule hooks) are accepted
        only for signature compatibility with other native models; passing
        them is rejected because the AOTI graph's JVP path is structurally
        different (it's a separate ``_jvp.pt2`` graph, accessed via the
        binding's ``jvp()`` / ``jacobian()`` methods rather than the typed
        chain-rule protocol). Drivers that don't need sensitivities -- e.g.
        ``TransientDriver``, ``TransientRegression`` -- work as-is.
        """
        if v is not None or v2 is not None or vh is not None:
            raise NotImplementedError(
                "AOTIModel does not support the native chain-rule v=/v2=/vh= "
                "arguments. Use the underlying binding's jvp() or jacobian() "
                "methods (self._inner) for sensitivities."
            )

        if len(args) != len(self.input_spec):
            raise TypeError(
                f"AOTIModel: expected {len(self.input_spec)} positional inputs "
                f"({list(self.input_spec)}), got {len(args)}."
            )

        # Unwrap typed wrappers to raw torch.Tensors and strictly validate
        # device + dtype against the artifact's pinned values. An AOTI .pt2
        # is compile-pinned to a specific (device, dtype) and the kernel
        # strides through input buffers assuming those exact widths; a wrong
        # device dereferences a host pointer in a CUDA kernel (illegal
        # access), and a wrong dtype reinterprets bytes (silent garbage at
        # small batches, SEGV at larger ones once the wrong-stride read
        # crosses past the mapped buffer end). We refuse to silently coerce
        # because that hides bugs in caller code and makes NEML2-as-library
        # surprising in larger pipelines -- the caller owns dtype/device
        # placement, mirroring v2's [Settings]-gated dtype policy. Compare
        # the v2 main()s (e.g. ``neml2-time.cxx`` calling
        # ``set_default_dtype(kFloat64)``) -- end applications set defaults,
        # the model boundary validates.
        #
        # Also record per-input ``sub_batch_ndim``: when the caller passes
        # a TensorWrapper, read it off the instance (the export-time
        # typed-wrapper trace baked the per-input split into the artifact,
        # so the caller's sub_batch_ndim is the authoritative source);
        # for raw torch.Tensor inputs, fall back to the metadata's
        # ``sub_batch_shape`` (the legacy path).
        raw_inputs: dict[str, torch.Tensor] = {}
        per_input_sub_ndim: dict[str, int] = {}
        for name, arg in zip(self.input_spec, args, strict=True):
            if isinstance(arg, TensorWrapper):
                t = arg.data
                per_input_sub_ndim[name] = arg.sub_batch_ndim
            elif isinstance(arg, torch.Tensor):
                t = arg
                per_input_sub_ndim[name] = len(self.input_sub_batch.get(name, ()))
            else:
                raise TypeError(
                    f"AOTIModel: input {name!r} must be a TensorWrapper or "
                    f"torch.Tensor, got {type(arg).__name__}."
                )
            self._check_tensor(t, name, kind="input")
            raw_inputs[name] = t

        # Also validate any promoted parameters the caller may have mutated
        # in place via ``self._inner.named_parameters()``. The .pt2 stores
        # them pinned to (target_device, target_dtype); if the user
        # overwrote with a tensor on a different device/dtype, the kernel
        # would silently dereference garbage.
        for pname, ptensor in self._inner.named_parameters().items():
            self._check_tensor(ptensor, pname, kind="parameter")

        # Normalize batch shapes. The .pt2 was traced with every input
        # carrying a single leading batch axis (``_example_inputs_for`` uses
        # ``zeros(2, *BASE_SHAPE)``) and torch.export installs a *shared*
        # dynamic Dim across all inputs -- they must agree on that dim at
        # runtime. Eager NEML2 callers freely pass base-only defaults (e.g.
        # ``TransientDriver._zero_for_step`` makes an SR2 input of shape
        # ``(6,)`` for unset history slots) and rely on the leaves'
        # TensorWrapper arithmetic to broadcast; the .pt2 graph has no such
        # broadcasting layer. Broadcast everything to the common batch
        # shape before crossing.
        raw_inputs, common_dyn = self._broadcast_to_common_batch(raw_inputs, per_input_sub_ndim)

        raw_outs = self._inner.forward(raw_inputs)
        # Wrap each output and recover ``sub_batch_ndim`` from the runtime
        # tensor shape. This is the same arithmetic v2's ``neml2::Tensor``
        # did at construction time:
        #
        #     out.shape == (*dyn, *sub, *base)
        #     dyn = common_dyn (broadcast across all inputs; outputs share)
        #     base = ``BASE_SHAPE`` from the typed wrapper class
        #     sub = whatever's left in the middle
        #
        # Solving for ``sub_n = out.ndim - dyn_n - BASE_NDIM``. Not a
        # heuristic -- there's exactly one structural decomposition that
        # fits. Without this step the default ``sub_batch_ndim=0`` wrap
        # mis-classifies a per-site axis as dyn, which breaks downstream
        # consumers that feed the output back as input (e.g.
        # ``TransientDriver.advance_step`` using step-N's per-crystal
        # output as step-(N+1)'s ``~1`` history).
        #
        # The assumption that output dyn == input common_dyn holds for
        # every shape-preserving forward operator NEML2 currently ships;
        # if a future model collapses batch axes inside the AOTI graph
        # (e.g. a reduction to a global summary), this would need to be
        # superseded by per-output sub_batch_shape carried in the
        # metadata. Cross that bridge when a model needs it.
        out_wrappers = []
        dyn_n = len(common_dyn)
        for name, type_cls in self.output_spec.items():
            raw = raw_outs[name]
            sub_n = max(raw.ndim - dyn_n - type_cls.BASE_NDIM, 0)
            # NOTE: sub_batch_labels are persisted in meta.json (see
            # ``output_labels``) but the static-base ``TensorWrapper``
            # subclasses do not accept a labels kwarg today, so the
            # re-attachment is a no-op. The shim still loads the labels
            # so that callers can introspect them (see
            # ``test_aoti_grain_label_round_trip``). Once labels become
            # load-bearing on static-base wrappers, set them here.
            out_wrappers.append(type_cls(raw, sub_batch_ndim=sub_n))
        return tuple(out_wrappers)

    def _broadcast_to_common_batch(
        self,
        raw_inputs: dict[str, torch.Tensor],
        per_input_sub_ndim: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor], torch.Size]:
        """Bring every input tensor to its declared ``(*dyn, *sub, *base)`` shape,
        with the dynamic-batch axes broadcast to a single common shape.

        Each input has three trailing-axis regions:

        * ``base`` -- ``BASE_SHAPE`` from the TensorWrapper class;
        * ``sub`` -- the per-input ``sub_batch_ndim`` resolved from the
          caller's typed wrapper instance (or the legacy metadata
          ``sub_batch_shape`` when the caller passes a raw tensor);
        * ``dyn`` -- everything in front, broadcast across all inputs
          (callers freely pass base-only defaults / single-step slices and
          rely on the runtime to lift them to the common dyn shape).

        The (dyn, sub, base) split is per-input -- a global Scalar with
        no sub-batch is reshaped to ``(*common_dyn, *base)`` while a
        per-crystal SR2 with ``sub_batch_ndim=1`` is reshaped to
        ``(*common_dyn, sub_size, *base)``. Without the per-input
        sub-batch carve-out, a naive ``torch.broadcast_shapes`` over the
        full batch region would collide the per-crystal axis against the
        global input's dyn axis.
        """
        dyn_shapes: list[torch.Size] = []
        for name, t in raw_inputs.items():
            base_ndim = self.input_spec[name].BASE_NDIM
            sub_ndim = per_input_sub_ndim[name]
            trail = base_ndim + sub_ndim
            dyn_shapes.append(t.shape if trail == 0 else t.shape[:-trail])
        common_dyn = torch.broadcast_shapes(*dyn_shapes)

        out: dict[str, torch.Tensor] = {}
        for name, t in raw_inputs.items():
            base_ndim = self.input_spec[name].BASE_NDIM
            base_shape = tuple(self.input_spec[name].BASE_SHAPE)
            sub_ndim = per_input_sub_ndim[name]
            # Read the sub-batch shape from the input tensor's current
            # trailing axes (the caller's wrapper already has the right
            # sub_batch_ndim; the raw tensor's trailing-axis sizes are
            # the sub-batch extents).
            sub_shape = (
                tuple(t.shape[len(t.shape) - base_ndim - sub_ndim : len(t.shape) - base_ndim])
                if sub_ndim
                else ()
            )
            target_shape = tuple(common_dyn) + sub_shape + base_shape
            if tuple(t.shape) == target_shape:
                out[name] = t
                continue
            view_shape = (1,) * (len(target_shape) - t.ndim) + tuple(t.shape)
            out[name] = t.view(view_shape).expand(target_shape).contiguous()
        return out, common_dyn
