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
/ ``input_base_shapes`` / ``output_base_shapes`` / ``device`` / ``dtype`` /
``forward`` / ``jvp`` / ``jacobian``).

Framework boundary (CLAUDE.md "Hard rules / Rule 1"): together with
:mod:`neml2._eager_boundary`, this module is the third legitimate raw-tensor
boundary (alongside the AOTI shim and ``torch.autograd.Function``). Raw
``torch.Tensor`` objects appear only in :meth:`_EagerModel.forward`'s argument
and return dicts, marshalled across the C++/Python line by torch's pybind
casters. They are wrapped into typed wrappers immediately on entry and unwrapped
back to raw only at the return boundary (in
:func:`neml2._eager_boundary.unwrap_outputs`).
"""

from __future__ import annotations

import itertools

import torch

from ._eager_boundary import (
    assemble_jacobian,
    assemble_jvp_outputs,
    broadcast_to_common_batch,
    check_tensor,
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
        from .cli.aoti_export import _var_infos
        from .factory import load_model
        from .models.common import ComposedModel

        model = load_model(input_file, model_name)
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
        variable's declared base shape, so an input/tangent is `(*B, *base)`
        (e.g. an SR2 is `(*B, 6)`, never `(*B, 1)` or `(*B, 3, 2)`). A Scalar has
        an empty base shape, so any leading shape is a valid batch.
        """
        base = tuple(self.input_spec[name].BASE_SHAPE)
        bn = len(base)
        trailing = tuple(t.shape[t.ndim - bn :]) if bn else ()
        if t.ndim < bn or trailing != base:
            raise ValueError(
                f"EagerModel {kind} {name!r}: non-canonical shape {tuple(t.shape)}; "
                f"expected trailing base shape {base} (canonical shape is (*B, *base))."
            )

    def _typed_args(self, inputs: dict[str, torch.Tensor]) -> tuple:
        """Validate + wrap the raw input dict into typed wrappers in input order.

        The inbound half of the raw-tensor boundary, shared by ``forward`` /
        ``jvp`` / ``jacobian``: checks for missing keys, validates each tensor's
        device + dtype, broadcasts to a common batch, and wraps each into its
        declared ``TensorWrapper`` (``sub_batch_ndim=0`` -- eager is a plain-batch
        runtime; sub-batched inputs are rejected on output, see ``_eval``).
        """
        missing = [n for n in self.input_names if n not in inputs]
        if missing:
            raise KeyError(f"EagerModel: missing input(s) {missing}; expected {self.input_names}.")
        # Eager inputs carry no baked sub-batch axes (plain-batch contract).
        per_input_sub_ndim = {n: 0 for n in self.input_names}
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
            raw[name] = t
        raw, _ = broadcast_to_common_batch(raw, self.input_spec, per_input_sub_ndim)
        return tuple(self.input_spec[name](raw[name]) for name in self.input_names)

    def _eval(self, typed_args: tuple, *, v=None) -> tuple[tuple, dict | None]:
        """Run the native model and enforce the plain-batch contract.

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
        self._reject_sub_batch(typed_outputs)
        return typed_outputs, v_out

    def _reject_sub_batch(self, typed_outputs: tuple) -> None:
        """Guard: the eager runtime is plain-batch only.

        A non-zero ``sub_batch_ndim`` on any output is the signal that the model
        carries BLOCK-aware / labelled axes (e.g. crystal-plasticity geometry),
        which the raw-tensor ``forward(dict)`` boundary cannot faithfully
        represent (per-input sub-batch shapes are caller-declared at AOTI compile
        time and have no slot here). Reject loudly instead of returning silently
        mislabelled results.
        """
        bad = [
            n
            for n, o in zip(self.output_names, typed_outputs, strict=True)
            if getattr(o, "sub_batch_ndim", 0) > 0
        ]
        if bad:
            raise NotImplementedError(
                f"EagerModel: output(s) {bad} carry sub-batch axes (sub_batch_ndim>0). "
                f"The eager runtime is plain-batch only; crystal-plasticity / "
                f"sub-batched models must use the AOTI / dispatched runtime."
            )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Evaluate the model on raw, name-keyed tensors.

        ``inputs`` is keyed by ``input_names``; missing keys raise ``KeyError``.
        Returns one raw tensor per name in ``output_names``. This is the
        raw-tensor boundary (see module docstring): inputs are validated and
        broadcast to a common batch, wrapped into typed wrappers, run through
        the native model, and unwrapped back to raw tensors at the return.
        """
        typed_outputs, _ = self._eval(self._typed_args(inputs))
        return unwrap_outputs(typed_outputs, self.output_names)

    def jvp(
        self,
        inputs: dict[str, torch.Tensor],
        tangents: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Evaluate the model and its Jacobian-vector product.

        ``inputs`` is keyed by ``input_names``; ``tangents`` shares those keys +
        ``(*B, *in_base)`` shapes, and a missing tangent defaults to zero (no
        contribution). Returns ``(outputs, jvp_outputs)`` -- both keyed by
        ``output_names``, mirroring the C++ ``neml2::aoti::Model::jvp`` contract;
        ``jvp_outputs[name]`` is the directional derivative at the output's
        natural ``(*batch, *out_base)``.

        Implemented on the native chain rule: each tangent is seeded as a single
        leading-K (K=1) direction; the per-input contributions to each output are
        summed at the boundary.
        """
        typed_args = self._typed_args(inputs)
        # All inputs share the common batch after broadcasting.
        common_batch = tuple(typed_args[0].batch_shape)
        seed: dict[str, dict] = {}
        for name in self.input_names:
            td = tangents.get(name)
            if td is None:
                continue
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
            td = torch.broadcast_to(td, (*common_batch, *base_shape))
            # Leading K=1 axis -> a single directional seed (see apply_chain_rule).
            seed[name] = {name: self.input_spec[name](td.unsqueeze(0))}
        typed_outputs, v_out = self._eval(typed_args, v=seed)
        outputs = unwrap_outputs(typed_outputs, self.output_names)
        assert v_out is not None
        jvp_outputs = assemble_jvp_outputs(v_out, typed_outputs, self.output_names)
        return outputs, jvp_outputs

    def jacobian(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Evaluate the model and its full Jacobian as variable-pair blocks.

        Returns ``(outputs, J)`` -- ``outputs`` keyed by ``output_names`` and ``J``
        the nested ``{out_name: {in_name: (*batch, *out_base, *in_base)}}`` block
        dict (rows in ``output_spec`` order, cols in ``input_spec`` order),
        matching the C++ ``neml2::aoti::Model::jacobian`` contract. Built on the
        native chain rule via a leading-K identity seed per input (reusing the
        AOTI export's ``_leading_k_identity_seed``).
        """
        from .cli.aoti_export import _leading_k_identity_seed

        typed_args = self._typed_args(inputs)
        seed = {
            name: {
                name: _leading_k_identity_seed(
                    self.input_spec[name],
                    typed_in.batch_shape,
                    dtype=typed_in.dtype,
                    device=typed_in.device,
                )
            }
            for name, typed_in in zip(self.input_names, typed_args, strict=True)
        }
        typed_outputs, v_out = self._eval(typed_args, v=seed)
        outputs = unwrap_outputs(typed_outputs, self.output_names)
        assert v_out is not None
        jac = assemble_jacobian(
            v_out,
            typed_outputs,
            self.output_names,
            self.output_spec,
            self.input_names,
            self.input_spec,
        )
        return outputs, jac
