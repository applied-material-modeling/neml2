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
        meta = './my_model_meta.json'
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
from ..schema import HitSchema, option
from ..types import TensorWrapper
from ._aoti import Model as _BoundModel

if TYPE_CHECKING:
    import nmhit

    from ..factory import _NativeInputFile


@register_neml2_object("AOTIModel")
class AOTIModel(nn.Module):
    """HIT-loadable wrapper around :class:`neml2.aoti.Model`.

    Constructed from a HIT ``[Models]`` block with a single ``meta`` option
    pointing at the metadata JSON produced by ``neml2-compile``. The path is
    resolved relative to the input file's directory.

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
            "meta",
            str,
            "Path to the AOTI metadata JSON produced by ``neml2-compile``. Resolved "
            "relative to the input file's directory when not absolute.",
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> AOTIModel:
        meta_rel = node.param_str("meta")
        # The meta path is HIT-style relative to the input file's directory
        # (mirrors how `neml2-compile` writes the stub: `./<name>_meta.json`).
        meta_path = (factory._path.parent / meta_rel).resolve()
        if not meta_path.exists():
            raise FileNotFoundError(
                f"AOTIModel({node.path()!r}): metadata file not found at "
                f"{meta_path} (HIT entry: meta={meta_rel!r}, resolved "
                f"relative to {factory._path})."
            )
        return cls(meta_path)

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

        raw_inputs = {}
        for name, arg in zip(self.input_spec, args, strict=True):
            if isinstance(arg, TensorWrapper):
                raw_inputs[name] = arg.data
            elif isinstance(arg, torch.Tensor):
                raw_inputs[name] = arg
            else:
                raise TypeError(
                    f"AOTIModel: input {name!r} must be a TensorWrapper or "
                    f"torch.Tensor, got {type(arg).__name__}."
                )

        raw_outs = self._inner.forward(raw_inputs)
        return tuple(self.output_spec[name](raw_outs[name]) for name in self.output_spec)
