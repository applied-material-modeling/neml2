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

"""Eager (uncompiled) model adapter for the C++ ``neml2::eager::Model`` runtime.

The C++ ``neml2::eager::Model`` (under ``neml2/csrc/eager/``) embeds a CPython
interpreter, imports this module, and constructs an :class:`_EagerModel` to run
a NEML2 model from C++ *without* the AOTI compile step. It is the
fast-to-start, slow-to-run sibling of the AOTI path: ideal for downstream C++
unit tests that cannot afford a minutes-long ``neml2-compile``.

Unlike :class:`neml2.aoti._shim.AOTIModel` (which is backed by a compiled
``.pt2`` and HIT-registered as ``type = AOTIModel``), this adapter is backed by
an ordinary Python-native model loaded via :func:`neml2.factory.load_model`
from the *original* ``.i`` file. It is not HIT-registered; it exists only to
present a raw-tensor, name-keyed surface to the C++ embed bridge that mirrors
the C++ ``neml2::aoti::Model`` pybind binding (``input_names`` / ``output_names``
/ ``input_base_shapes`` / ``output_base_shapes`` / ``parameter_base_shapes`` /
``device`` / ``dtype`` / ``forward`` / ``jvp`` / ``jacobian``).

Framework boundary (CLAUDE.md "Hard rules / Rule 1"): together with
:mod:`neml2.types._boundary`, this module is the third legitimate raw-tensor
boundary (alongside the AOTI shim and ``torch.autograd.Function``). Raw
``torch.Tensor`` objects appear only in :meth:`_EagerModel.forward`'s argument
and return dicts, marshalled across the C++/Python line by torch's pybind
casters. They are wrapped into typed wrappers immediately on entry and unwrapped
back to raw only at the return boundary (in
:func:`neml2.types._boundary.unwrap_outputs`).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch

from ..types._boundary import (
    assemble_jacobian,
    assemble_jvp_outputs,
    broadcast_to_common_batch,
    check_tensor,
    sub_batched_identity_seed,
    unwrap_outputs,
)

#: ``_EagerModel`` is the entry point the C++ ``neml2::eager::Model`` imports and
#: constructs; nothing in the Python package references it, so name it in
#: ``__all__`` to mark it as the module's public surface.
__all__ = ["_EagerModel"]


def _infer_device_dtype(
    model: torch.nn.Module,
    device_override: str | torch.device | None,
) -> tuple[torch.device, torch.dtype]:
    """Resolve the model's working ``(device, dtype)``.

    dtype is read from the model's first floating-point parameter / buffer (NEML2
    builds typed parameters at float64 regardless of the torch default, so this
    is the authoritative source); device is the override when given, else the
    model's tensor device, else the torch default. Falls back to the torch
    defaults for a model that carries no tensors at all.
    """
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for t in itertools.chain(model.parameters(), model.buffers()):
        if t.is_floating_point():
            dtype = t.dtype
            device = t.device
            break
        if device is None:
            device = t.device
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device_override is not None:
        device = torch.device(device_override)
    elif device is None:
        device = torch.get_default_device()
    return device, dtype


class _EagerModel:
    """Raw-tensor, name-keyed adapter around a Python-native NEML2 model.

    Parameters
    ----------
    input_file:
        Path to the original HIT ``.i`` file (the same one ``neml2-run`` /
        :func:`neml2.factory.load_model` consume -- *not* a compiled stub).
    model_name:
        Name of the model in the file's ``[Models]`` section.
    device:
        Optional device override (a string like ``"cpu"`` / ``"cuda:0"`` or a
        ``torch.device``). When given, the model is moved there; when omitted,
        the model's natural device is used.
    """

    def __init__(
        self,
        input_file: str,
        model_name: str,
        device: str | torch.device | None = None,
    ) -> None:
        # Lazy imports: keep module import (which the C++ ctor triggers) cheap
        # and avoid any import-time coupling to the CLI / model packages.
        from ..cli.aoti_export import _var_infos, plan_output_sub_batch_shapes
        from ..factory import load_input
        from ..models.common import ComposedModel
        from ..settings import model_variable_names

        factory = load_input(input_file)
        model = factory.get_model(model_name)
        # `[Settings]/example_batch_shape` is optional. When present it is checked
        # here, as neml2-compile checks it: an unknown name raises at load.
        settings = factory.settings
        settings.validate_names(
            model_variable_names(model), universe=f"{type(model).__name__} variables"
        )
        # Wrap a bare leaf in a ComposedModel for a stable plain-tensor boundary,
        # exactly as the AOTI export path does (see neml2/cli/aoti_export.py).
        self._model = model if isinstance(model, ComposedModel) else ComposedModel([model])
        if device is not None:
            self._model = self._model.to(torch.device(device))
        self._device, self._dtype = _infer_device_dtype(self._model, device)

        # input/output names + base shapes via the SAME helper the AOTI metadata
        # path uses (_var_infos). This is the parity guarantee: an eager model
        # reports byte-identical names/base-shapes to its AOTI-compiled twin, so
        # the C++ neml2::eager::Model is a true drop-in for neml2::aoti::Model.
        self.input_spec = self._model.input_spec
        self.output_spec = self._model.output_spec
        in_infos = _var_infos(self.input_spec)
        out_infos = _var_infos(self.output_spec)
        self.input_names: list[str] = [i["name"] for i in in_infos]
        self.output_names: list[str] = [i["name"] for i in out_infos]
        self.input_base_shapes: list[list[int]] = [list(i["base_shape"]) for i in in_infos]
        self.output_base_shapes: list[list[int]] = [list(i["base_shape"]) for i in out_infos]

        # Sub-batch extents. This runtime reports *declared* extents: an empty
        # entry means "not declared as sub-batched", not "plain batch". An input's
        # extent is its `[Settings]` declaration (None when undeclared). Output
        # extents come from the same plan neml2-compile records them from, so an
        # eager model reports what its AOTI twin would; the plan runs only when
        # the file declares a sub-batch extent, since without one there is
        # nothing to report.
        self._declared_sub: dict[str, tuple[int, ...] | None] = {
            n: settings.sub_batch_shape(n) for n in self.input_names
        }
        self.input_sub_batch_shapes: list[list[int]] = [
            list(self._declared_sub[n] or ()) for n in self.input_names
        ]
        declares_sub = any(
            settings.sub_batch_shape(n) is not None for n in model_variable_names(model)
        )
        planned: dict[str, list[int]] = {}
        if declares_sub:
            # The extents are authoritative, so a planner failure fails the load;
            # say where it came from, since the model itself may evaluate fine.
            try:
                planned = plan_output_sub_batch_shapes(
                    input_file, model_name, device=str(self._device)
                )
            except Exception as err:
                raise RuntimeError(
                    f"EagerModel: could not work out the output sub-batch shapes for the "
                    f"[Settings]/example_batch_shape declaration of model {model_name!r} in "
                    f"'{input_file}' (the compile plan failed: {type(err).__name__}: {err})."
                ) from err
        self.output_sub_batch_shapes: list[list[int]] = [
            list(planned.get(n, [])) for n in self.output_names
        ]

        # Calibration-parameter surface (d(output)/d(parameter), reverse-mode AD).
        # Parameters are addressed by their named_parameters() qualified name
        # (e.g. "elasticity.E"); base shapes come from each parameter's typed
        # wrapper class. This is the parameter analogue of the input surface and
        # is what the C++ neml2::eager::Model consumes for param_jacobian.
        from ..models.param_ad import enumerate_typed_params

        self._typed_params = enumerate_typed_params(self._model)
        # Internal name list feeding the reverse-mode AD engine (param_jacobian /
        # param_vjp / named_parameters); the engine returns name-keyed dicts, so its
        # order is irrelevant. The public surface is the qualified-name -> base shape
        # map below (unified with aoti::Model.parameter_base_shapes).
        self._param_names: list[str] = [q for q, _ in self._typed_params]
        # {qname: natural base shape}; used to split a (possibly batched) parameter's
        # stored shape into (batch, base) when computing the call batch / param blocks.
        self._param_base_shape_map: dict[str, tuple[int, ...]] = {
            q: tuple(tc.BASE_SHAPE) for q, tc in self._typed_params
        }
        # Public parameter-base-shape map keyed by qualified name (Scalar -> [],
        # SR2 -> [6]); the parameter analogue of input_base_shapes and the unified
        # surface the C++ neml2::eager::Model / neml2::aoti::Model expose. Keys are
        # the calibration parameters (same keys as named_parameters()).
        self.parameter_base_shapes: dict[str, list[int]] = {
            q: list(b) for q, b in self._param_base_shape_map.items()
        }

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def _device_hint(self) -> str:
        return (
            f"The model runs on (device={self._device}, dtype={self._dtype}). "
            f"Cast the tensor explicitly with ``.to(device=..., dtype=...)`` at "
            f"the call site before evaluating."
        )

    def _check_canonical(self, name: str, t: torch.Tensor, *, kind: str) -> None:
        """Reject a non-canonical tensor: its trailing axes must equal the
        variable's declared base shape (e.g. an SR2 ends in `6`, never `1` or
        `(3, 2)`). Only the trailing base is checked here; the leading axes are
        `(*B, *sub_batch)` for an input and `(*B,)` for a tangent, split by
        :meth:`_sub_batch_ndim`. A Scalar has an empty base shape, so any
        leading shape passes.
        """
        base = tuple(self.input_spec[name].BASE_SHAPE)
        bn = len(base)
        trailing = tuple(t.shape[t.ndim - bn :]) if bn else ()
        if t.ndim < bn or trailing != base:
            raise ValueError(
                f"EagerModel {kind} {name!r}: non-canonical shape {tuple(t.shape)}; "
                f"expected trailing base shape {base} (canonical shape is (*B, *sub_batch, *base))."
            )

    def _sub_batch_ndim(self, name: str, t: torch.Tensor, batch_ndim: int | None) -> int:
        """Number of sub-batch axes of input *name* passed as *t*.

        Every axis between the leading *batch_ndim* batch axes and the base shape
        is sub-batch. Without *batch_ndim*, the input's ``[Settings]``
        declaration gives the count, and an undeclared input is plain batch. A
        declared extent is checked against the tensor either way, as AOTI checks
        its compiled metadata; a size-1 axis broadcasts to the declared extent.
        """
        base_ndim = len(self.input_spec[name].BASE_SHAPE)
        declared = self._declared_sub[name]
        if batch_ndim is not None:
            # Clamp, not error: an input with fewer leading axes than batch_ndim
            # (e.g. a base-only 0-d time) has no sub-batch and broadcasts.
            n = max(t.ndim - base_ndim - batch_ndim, 0)
        elif declared is not None:
            n = len(declared)
        else:
            return 0
        if declared is not None:
            sub = tuple(t.shape[t.ndim - base_ndim - n : t.ndim - base_ndim])
            if (
                t.ndim < base_ndim + n
                or len(sub) != len(declared)
                or any(s not in (d, 1) for s, d in zip(sub, declared, strict=True))
            ):
                raise ValueError(
                    f"EagerModel input {name!r}: shape {tuple(t.shape)} does not carry the "
                    f"sub-batch extent {declared} declared in [Settings]/example_batch_shape "
                    f"(canonical shape is (*B, *sub_batch, *base)"
                    + ("" if batch_ndim is None else f", with {batch_ndim} batch axes")
                    + ")."
                )
        return n

    def _typed_args(
        self, inputs: dict[str, torch.Tensor], batch_ndim: int | None = None
    ) -> tuple[tuple, torch.Size]:
        """Validate + wrap the raw input dict into typed wrappers in input order.

        The inbound half of the raw-tensor boundary, shared by ``forward`` /
        ``jvp`` / ``jacobian``: checks for missing keys, validates each tensor's
        device + dtype, splits it into ``(*B, *sub_batch, *base)`` (see
        :meth:`_sub_batch_ndim`), broadcasts the batch axes to a common batch and
        each sub-batch axis to its declared extent, and wraps each into its
        declared ``TensorWrapper`` with that ``sub_batch_ndim``. Returns the
        typed inputs and the common batch.
        """
        missing = [n for n in self.input_names if n not in inputs]
        if missing:
            raise KeyError(f"EagerModel: missing input(s) {missing}; expected {self.input_names}.")
        if batch_ndim is not None and batch_ndim < 0:
            raise ValueError(f"EagerModel: batch_ndim must be non-negative, got {batch_ndim}.")
        per_input_sub_ndim: dict[str, int] = {}
        raw: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            t = inputs[name]
            check_tensor(
                t,
                name,
                self._device,
                self._dtype,
                kind="input",
                context="EagerModel",
                hint=self._device_hint,
            )
            self._check_canonical(name, t, kind="input")
            n = self._sub_batch_ndim(name, t, batch_ndim)
            declared = self._declared_sub[name]
            if n and declared is not None:
                # Lift any size-1 sub-batch axis to its declared extent.
                base = tuple(self.input_spec[name].BASE_SHAPE)
                t = t.expand(*t.shape[: t.ndim - len(base) - n], *declared, *base)
            per_input_sub_ndim[name] = n
            raw[name] = t
        raw, common_dyn = broadcast_to_common_batch(raw, self.input_spec, per_input_sub_ndim)
        typed = tuple(
            self.input_spec[name](raw[name], sub_batch_ndim=per_input_sub_ndim[name])
            for name in self.input_names
        )
        return typed, common_dyn

    def _plain_inputs(self, typed_args: tuple) -> list[str]:
        """Names of the inputs that carry no sub-batch axes in this call.

        The default derivative columns; a sub-batched input is included only
        when named.
        """
        return [
            n for n, ta in zip(self.input_names, typed_args, strict=True) if ta.sub_batch_ndim == 0
        ]

    def _reject_sub_batch(self, names: list[str], wrappers: tuple, what: str) -> None:
        """Guard for the parameter-derivative paths, which are plain-batch only.

        ``forward`` / ``jvp`` / ``jacobian`` accept sub-batched variables; the
        reverse-mode parameter derivatives have not been extended to them, so a
        call that carries one is rejected instead of returning mis-shaped blocks.
        Called on the inputs before the evaluation (so a rejected call pays no
        solve) and on the outputs after it (a plain input can still produce a
        sub-batched output).
        """
        bad = [n for n, w in zip(names, wrappers, strict=True) if w.sub_batch_ndim > 0]
        if bad:
            raise NotImplementedError(
                f"EagerModel {what}: {bad} carry sub-batch axes; parameter derivatives "
                f"are plain-batch only on this runtime."
            )

    def _eval(self, typed_args: tuple, *, v=None) -> tuple[tuple, dict | None]:
        """Run the native model.

        Returns ``(typed_outputs, v_out)`` -- ``v_out`` is ``None`` when ``v`` is
        not supplied. ``ComposedModel.forward`` returns a bare output tuple when
        ``v`` is ``None`` and ``(*outputs, v_out)`` when ``v`` is given.
        """
        result = self._model(*typed_args, v=v)
        if v is None:
            typed_outputs = result if isinstance(result, tuple) else (result,)
            v_out = None
        else:
            *outs, v_out = result
            typed_outputs = tuple(outs)
        return typed_outputs, v_out

    def forward(
        self, inputs: dict[str, torch.Tensor], batch_ndim: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Evaluate the model on raw, name-keyed tensors.

        ``inputs`` is keyed by ``input_names``; missing keys raise ``KeyError``.
        Each tensor is ``(*B, *sub_batch, *base)``. *batch_ndim*, when given, is
        the number of leading batch axes ``B``; otherwise an input's
        ``[Settings]`` declaration gives its sub-batch axes, and an undeclared
        input is plain batch. Returns one raw tensor per name in
        ``output_names`` at ``(*B, *out_sub_batch, *out_base)``, contiguous.
        This is the raw-tensor boundary (see module docstring): inputs are
        validated and broadcast to a common batch, wrapped into typed wrappers,
        run through the native model, and unwrapped back to raw tensors at the
        return.
        """
        typed_args, common_dyn = self._typed_args(inputs, batch_ndim)
        typed_outputs, _ = self._eval(typed_args)
        return unwrap_outputs(typed_outputs, self.output_names, common_dyn)

    def jvp(
        self,
        inputs: dict[str, torch.Tensor],
        tangents: dict[str, torch.Tensor],
        batch_ndim: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Evaluate the model and its Jacobian-vector product.

        ``inputs`` is keyed by ``input_names`` and split as in :meth:`forward`;
        ``tangents`` shares those keys + ``(*B, *in_base)`` shapes, and a missing
        tangent defaults to zero (no contribution). A tangent for a sub-batched
        input is rejected: derivatives are taken with respect to plain-batch
        inputs only. Returns ``(outputs, jvp_outputs)`` -- both keyed by
        ``output_names``, mirroring the C++ ``neml2::aoti::Model::jvp`` contract;
        ``jvp_outputs[name]`` is the directional derivative, materialized like
        ``outputs[name]`` at ``(*B, *out_sub_batch, *out_base)``, contiguous.

        Implemented on the native chain rule: each tangent is seeded as a single
        leading-K (K=1) direction; the per-input contributions to each output are
        summed at the boundary.
        """
        typed_args, common_dyn = self._typed_args(inputs, batch_ndim)
        plain = set(self._plain_inputs(typed_args))
        seed: dict[str, dict] = {}
        for name in self.input_names:
            td = tangents.get(name)
            if td is None:
                continue
            if name not in plain:
                raise ValueError(
                    f"EagerModel jvp: input {name!r} is sub-batched; derivatives are taken "
                    f"with respect to plain-batch inputs only."
                )
            check_tensor(
                td,
                name,
                self._device,
                self._dtype,
                kind="tangent",
                context="EagerModel",
                hint=self._device_hint,
            )
            self._check_canonical(name, td, kind="tangent")
            base_shape = tuple(self.input_spec[name].BASE_SHAPE)
            td = torch.broadcast_to(td, (*common_dyn, *base_shape))
            # Leading K=1 axis -> a single directional seed (see apply_chain_rule).
            seed[name] = {name: self.input_spec[name](td.unsqueeze(0))}
        typed_outputs, v_out = self._eval(typed_args, v=seed)
        outputs = unwrap_outputs(typed_outputs, self.output_names, common_dyn)
        assert v_out is not None
        jvp_outputs = assemble_jvp_outputs(v_out, typed_outputs, self.output_names, common_dyn)
        return outputs, jvp_outputs

    def jacobian(
        self,
        inputs: dict[str, torch.Tensor],
        batch_ndim: int | None = None,
        derivatives: Sequence[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Evaluate the model and its Jacobian as variable-pair blocks.

        ``inputs`` is split as in :meth:`forward`. Returns ``(outputs, J)`` with
        ``J`` the nested ``{out_name: {in_name: block}}`` dict, matching the C++
        ``neml2::aoti::Model::jacobian`` contract.

        *derivatives* takes ``neml2-compile -d`` specs; the default is every output
        against the plain-batch inputs. A sub-batched input is included only when
        named (block forms: :func:`neml2.types._boundary.sub_batched_pair_block`).
        """
        from ..cli.aoti_export import _leading_k_identity_seed, _resolve_derivative_specs
        from ..models.param_ad import call_batch_shape

        typed_args, common_dyn = self._typed_args(inputs, batch_ndim)
        plain = self._plain_inputs(typed_args)
        sub_shapes = {
            name: tuple(int(s) for s in ta.sub_batch_shape)
            for name, ta in zip(self.input_names, typed_args, strict=True)
            if name not in plain
        }
        if derivatives is None:
            pairs = None
            columns = plain
        else:
            pairs, _ = _resolve_derivative_specs(
                derivatives, self.output_names, self.input_names, sub_batched_inputs=sub_shapes
            )
            requested = {i for _, i in pairs}
            columns = [n for n in self.input_names if n in requested]
        # The output batches on broadcast(input batches, parameter batches) -- a
        # parameter may carry its own batch dim (e.g. a per-batch-element Scalar).
        # Build every input's identity seed at that common call batch so the chain
        # rule and the (output-batch-keyed) assembly agree; a per-input batch would
        # mismatch whenever a parameter's batch exceeds the inputs'.
        call_batch = call_batch_shape(
            typed_args, self._model, self._param_names, self._param_base_shape_map
        )
        seed = {}
        for name, typed_in in zip(self.input_names, typed_args, strict=True):
            if name not in columns:
                continue
            opts = {"dtype": typed_in.dtype, "device": typed_in.device}
            if name in sub_shapes:
                leaf = sub_batched_identity_seed(
                    self.input_spec[name], call_batch, sub_shapes[name], **opts
                )
            else:
                leaf = _leading_k_identity_seed(self.input_spec[name], call_batch, **opts)
            seed[name] = {name: leaf}
        typed_outputs, v_out = self._eval(typed_args, v=seed)
        outputs = unwrap_outputs(typed_outputs, self.output_names, common_dyn)
        assert v_out is not None
        jac = assemble_jacobian(
            v_out,
            typed_outputs,
            self.output_names,
            self.output_spec,
            columns,
            self.input_spec,
            call_batch,
            input_sub_batch_shapes=sub_shapes,
            pairs=pairs,
        )
        return outputs, jac

    def named_parameters(self) -> dict[str, torch.Tensor]:
        """Raw, name-keyed view of the model's calibration parameters.

        Keyed by the qualified ``named_parameters()`` name (e.g. ``"elasticity.E"``);
        values are detached raw tensors at the parameter's natural ``(*param_base)``
        shape. This is the read side of the parameter surface the C++
        ``neml2::eager::Model`` exposes (the value analogue of
        :attr:`parameter_base_shapes`); it is part of the raw-tensor embed boundary
        (see module docstring).
        """
        params = dict(self._model.named_parameters())
        return {q: params[q].detach() for q in self._param_names}

    def set_parameter(self, name: str, value: torch.Tensor) -> None:
        """Replace a calibration parameter's value -- the write side of
        :meth:`named_parameters`, mirroring ``neml2::eager::Model::set_parameter``
        / the AOTI ``set_parameter``. Forwards to the native model's
        :meth:`~neml2.models.model.Model.set_parameter` (a ``torch.no_grad``
        in-place copy into the live ``nn.Parameter``). *value* is a raw tensor at
        the parameter's ``(*param_base)`` shape (raw-tensor embed boundary)."""
        self._model.set_parameter(name, value)

    def param_jacobian(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Evaluate the model and its Jacobian w.r.t. the calibration parameters.

        Returns ``(outputs, P)`` -- ``outputs`` keyed by ``output_names`` and ``P``
        the nested ``{out_name: {param_qname: (*batch, *out_base, *param_base)}}``
        block dict (rows in ``output_spec`` order, columns keyed by the qualified
        parameter name), the parameter analogue of :meth:`jacobian`'s variable-pair
        output.

        Unlike :meth:`jacobian` (forward-mode input chain rule), this uses
        reverse-mode autograd over the model's parameters via
        :func:`neml2.models.param_ad.param_jacobian`; the input ``v=`` path is not
        touched. Composition through an ``ImplicitUpdate`` (Newton solve) is
        handled by that function's reverse pass (the implicit-function-theorem
        adjoint in ``_ImplicitUpdateFn.backward``). A constant ``(out, param)``
        pair is an explicit zero block.
        """
        from ..models.param_ad import param_jacobian as _param_jacobian

        typed_args, common_dyn = self._typed_args(inputs)
        self._reject_sub_batch(self.input_names, typed_args, "param_jacobian")
        # Plain forward for the value outputs + the output half of the guard; the
        # parameter Jacobian re-runs the forward under functional_call.
        typed_outputs, _ = self._eval(typed_args)
        self._reject_sub_batch(self.output_names, typed_outputs, "param_jacobian")
        outputs = unwrap_outputs(typed_outputs, self.output_names, common_dyn)
        pjac = _param_jacobian(
            self._model,
            typed_args,
            self._param_names,
            self.output_names,
            self.output_spec,
            self._param_base_shape_map,
        )
        return outputs, pjac

    def param_vjp(
        self,
        inputs: dict[str, torch.Tensor],
        cotangents: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        r"""Parameter adjoint ``dL/d\theta`` for ``L = sum_o <cotangent_o, out_o>``.

        Returns ``{param_qname: grad}`` at each parameter's natural shape (a
        scalar parameter yields a scalar, the batch summed out). One reverse pass
        total via :func:`neml2.models.param_ad.param_vjp` -- the cheap form for
        many-parameter (inverse-optimization) gradients, and the eager analogue
        of the compiled ``param_vjp``. Composition through an ``ImplicitUpdate``
        is handled by the implicit-function-theorem adjoint in the parameter
        swap's backward, exactly as in :meth:`param_jacobian`.

        *cotangents* maps each output name to ``w_o`` at the output's
        ``(*B, *out_base)`` shape (raw tensor). A call whose inputs or outputs
        carry sub-batch axes raises ``NotImplementedError``.
        """
        from ..models.param_ad import param_vjp as _param_vjp

        typed_args, _ = self._typed_args(inputs)
        self._reject_sub_batch(self.input_names, typed_args, "param_vjp")
        # Plain forward runs the output half of the guard before the adjoint.
        typed_outputs, _ = self._eval(typed_args)
        self._reject_sub_batch(self.output_names, typed_outputs, "param_vjp")
        return _param_vjp(
            self._model,
            typed_args,
            self._param_names,
            self.output_names,
            cotangents,
        )
