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
from ..types._boundary import broadcast_to_common_batch, check_tensor
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
    back in its declared type. Promoted parameters (if any) are *not* part of
    ``input_spec``; they are reachable through :meth:`named_parameters` (a
    mutable dict) and :meth:`set_parameter`. The full sensitivity surface --
    :meth:`jvp`, :meth:`jacobian`, :meth:`param_jacobian`, :meth:`param_vjp` --
    is forwarded to the binding so the py-aoti route matches the others
    (CLAUDE.md "six evaluation routes" parity).
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

        Thin wrapper over the shared
        :func:`neml2.types._boundary.check_tensor` (the device/dtype
        validation is identical across the AOTI + eager boundaries). ``kind``
        is ``'input'`` or ``'parameter'``; the AOTI-specific remediation
        guidance is supplied here as the ``hint``.
        """
        target_device = self._inner.device
        target_dtype = self._inner.dtype
        hint = (
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
        check_tensor(
            t, name, target_device, target_dtype, kind=kind, context="AOTIModel", hint=hint
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
        different (it's a separate ``_jvp.pt2`` graph, exposed via the
        :meth:`jvp` / :meth:`jacobian` methods rather than the typed
        chain-rule protocol). Drivers that don't need sensitivities -- e.g.
        ``TransientDriver``, ``TransientRegression`` -- work as-is.
        """
        if v is not None or v2 is not None or vh is not None:
            raise NotImplementedError(
                "AOTIModel does not support the native chain-rule v=/v2=/vh= "
                "arguments. Use the jvp() / jacobian() methods for sensitivities "
                "(the AOTI derivative graph is a separate compiled artifact)."
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

    def jvp(
        self,
        inputs: dict[str, torch.Tensor],
        tangents: dict[str, torch.Tensor],
        param_overrides: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Outputs + the directional derivative ``J @ v`` from the compiled
        ``jvp`` graph.

        ``inputs`` / ``tangents`` are ``{variable_name: tensor}`` dicts (same
        keys as :attr:`input_spec`); ``tangents`` carries the seed direction
        per input. Returns ``(outputs, jvp)`` keyed by output name -- the same
        raw-tensor surface the eager runtime exposes. Requires the artifact to
        carry the derivative graph (compiled with ``neml2-compile ... -d
        OUT:IN``); otherwise the underlying binding raises.
        """
        return self._inner.jvp(inputs, tangents, param_overrides or {})

    def jacobian(
        self,
        inputs: dict[str, torch.Tensor],
        param_overrides: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Outputs + the per-(output, input) Jacobian blocks from the compiled
        ``jacobian`` graph.

        ``inputs`` is a ``{variable_name: tensor}`` dict. Returns
        ``(outputs, J)`` where ``J[out][in]`` is the block ``d(out)/d(in)``,
        shaped ``(batch, *out_base, *in_base)``. Like :meth:`jvp`, needs an
        artifact compiled with ``-d``.
        """
        return self._inner.jacobian(inputs, param_overrides or {})

    def param_jacobian(
        self,
        inputs: dict[str, torch.Tensor],
        param_overrides: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Outputs + the per-(output, parameter) Jacobian blocks ``d(out)/d(param)``.

        The reverse-mode parameter-derivative counterpart to :meth:`jacobian`
        (mirrors :meth:`neml2.models.model.Model.param_jacobian` on the other
        routes): ``inputs`` is a ``{variable_name: tensor}`` dict and the result
        is ``(outputs, P)`` with ``P[out][param]`` the block ``d(out)/d(param)``,
        keyed by qualified promoted-parameter name (see :meth:`named_parameters`).
        Needs an artifact with the parameter promoted (``neml2-compile ... -p
        NAME``) and its derivative graph (``-d``).
        """
        return self._inner.param_jacobian(inputs, param_overrides or {})

    def param_vjp(
        self,
        inputs: dict[str, torch.Tensor],
        cotangents: dict[str, torch.Tensor],
        param_overrides: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reverse-mode parameter VJP -- ``cotangent`` contracted with ``d(out)/d(param)``.

        The reverse-mode counterpart to :meth:`param_jacobian` (mirrors
        :meth:`neml2.models.model.Model.param_vjp`): ``inputs`` is a
        ``{variable_name: tensor}`` dict, ``cotangents`` is keyed by output name,
        and the result is keyed by qualified promoted-parameter name. Needs a
        ``-p`` / ``-d`` artifact.
        """
        return self._inner.param_vjp(inputs, cotangents, param_overrides or {})

    def named_parameters(self) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """The promoted parameters as a mutable ``{qualified_name: tensor}`` dict.

        Overrides ``nn.Module.named_parameters`` -- whose ``(name, Parameter)``
        iterator would be empty here, since the shim registers no
        ``nn.Parameter``s of its own (the calibratable values live in the
        compiled binding). This returns the binding's promoted-parameter surface,
        the same dict the eager runtime and the C++ routes expose. Entries may be
        mutated in place (``m.named_parameters()["elasticity.E"].fill_(...)``) or
        replaced via :meth:`set_parameter`; the change is seen on the next call.
        Only parameters promoted at compile time (``neml2-compile ... -p NAME``)
        appear -- with none promoted the dict is empty.
        """
        return self._inner.named_parameters()

    def set_parameter(self, name: str, value: torch.Tensor) -> None:
        """Replace a promoted parameter's value (forwarded to the binding).

        ``name`` is a qualified promoted-parameter name (a key of
        :meth:`named_parameters`); ``value`` must match the artifact's pinned
        device/dtype. The new value is used on the next forward / derivative call.
        """
        self._inner.set_parameter(name, value)

    def _broadcast_to_common_batch(
        self,
        raw_inputs: dict[str, torch.Tensor],
        per_input_sub_ndim: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor], torch.Size]:
        """Bring every input tensor to its declared ``(*dyn, *sub, *base)`` shape,
        with the dynamic-batch axes broadcast to a single common shape.

        Thin wrapper over the shared
        :func:`neml2.types._boundary.broadcast_to_common_batch`; see there for
        the (dyn, sub, base) per-input split rationale. The per-input
        ``sub_batch_ndim`` is resolved from the caller's typed wrapper instance
        (or the legacy metadata ``sub_batch_shape`` when the caller passes a
        raw tensor) before this call.
        """
        return broadcast_to_common_batch(raw_inputs, self.input_spec, per_input_sub_ndim)
