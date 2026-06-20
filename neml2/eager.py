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
/ ``input_sizes`` / ``output_sizes`` / ``device`` / ``dtype`` /
``forward(dict) -> dict``).

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

from ._eager_boundary import broadcast_to_common_batch, check_tensor, unwrap_outputs

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

        # input/output names + flat sizes via the SAME helper the AOTI metadata
        # path uses (_var_infos). This is the parity guarantee: an eager model
        # reports byte-identical names/sizes to its AOTI-compiled twin, so the
        # C++ neml2::eager::Model is a true drop-in for neml2::aoti::Model.
        self.input_spec = self._model.input_spec
        self.output_spec = self._model.output_spec
        in_infos = _var_infos(self.input_spec)
        out_infos = _var_infos(self.output_spec)
        self.input_names: list[str] = [i["name"] for i in in_infos]
        self.output_names: list[str] = [i["name"] for i in out_infos]
        self.input_sizes: list[int] = [int(i["var_size"]) for i in in_infos]
        self.output_sizes: list[int] = [int(i["var_size"]) for i in out_infos]

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Evaluate the model on raw, name-keyed tensors.

        ``inputs`` is keyed by ``input_names``; missing keys raise ``KeyError``.
        Returns one raw tensor per name in ``output_names``. This is the
        raw-tensor boundary (see module docstring): inputs are validated and
        broadcast to a common batch, wrapped into typed wrappers, run through
        the native model, and unwrapped back to raw tensors at the return.
        """
        missing = [n for n in self.input_names if n not in inputs]
        if missing:
            raise KeyError(
                f"EagerModel.forward: missing input(s) {missing}; expected {self.input_names}."
            )
        hint = (
            f"The model runs on (device={self._device}, dtype={self._dtype}). "
            f"Cast the tensor explicitly with ``.to(device=..., dtype=...)`` at "
            f"the call site before evaluating."
        )
        # Eager inputs carry no baked sub-batch axes (the forward-only simple-test
        # use case). Crystal-plasticity sub-batch inputs are out of scope for now.
        per_input_sub_ndim = {n: 0 for n in self.input_names}
        raw: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            t = inputs[name]
            check_tensor(
                t, name, self._device, self._dtype, kind="input", context="EagerModel", hint=hint
            )
            raw[name] = t
        raw, _ = broadcast_to_common_batch(raw, self.input_spec, per_input_sub_ndim)

        # Wrap to typed at the boundary, run the native model, unwrap to raw.
        typed_args = tuple(self.input_spec[name](raw[name]) for name in self.input_names)
        typed_outs = self._model(*typed_args)
        if not isinstance(typed_outs, tuple):
            typed_outs = (typed_outs,)
        return unwrap_outputs(typed_outs, self.output_names)
