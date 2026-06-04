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

"""Python-native mirror of C++ ``common/ConstantExtrapolationPredictor.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, option
from ...types import (
    R2,
    SR2,
    Rot,
    Scalar,
    TensorWrapper,
)
from .._hit import _opt_list_str


def _read_list_str(node, name):  # noqa: ANN001, ANN202
    return list(node.param_list_str(name))


@register_neml2_object("ConstantExtrapolationPredictor")
class ConstantExtrapolationPredictor(Model):
    """Initial guess for an implicit update: each unknown takes its ``~1`` value.

    For an ``ImplicitUpdate`` with unknowns ``{u_i}``, this predictor reads
    ``u_i~1`` and outputs $u_i$. Used by ``ImplicitUpdate`` to seed Newton.
    Matches the C++ ``ConstantExtrapolationPredictor`` HIT signature.
    """

    hit = HitSchema(
        option(
            "unknowns_SR2",
            list,
            "The unknowns to extrapolate of type SR2",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_Scalar",
            list,
            "The unknowns to extrapolate of type Scalar",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_Rot",
            list,
            "The unknowns to extrapolate of type Rot",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_R2",
            list,
            "The unknowns to extrapolate of type R2",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
    )

    def __init__(
        self,
        unknowns_SR2: list[str],
        unknowns_Scalar: list[str],
        unknowns_Rot: list[str] | None = None,
        unknowns_R2: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._sr2 = list(unknowns_SR2)
        self._scalar = list(unknowns_Scalar)
        self._rot = list(unknowns_Rot or [])
        self._r2 = list(unknowns_R2 or [])
        self.input_spec = {
            **{f"{u}~1": SR2 for u in self._sr2},
            **{f"{u}~1": Scalar for u in self._scalar},
            **{f"{u}~1": Rot for u in self._rot},
            **{f"{u}~1": R2 for u in self._r2},
        }
        self.output_spec = {
            **{u: SR2 for u in self._sr2},
            **{u: Scalar for u in self._scalar},
            **{u: Rot for u in self._rot},
            **{u: R2 for u in self._r2},
        }

    def forward(  # type: ignore[override]
        self,
        *inputs: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        # Predictor is a pure pass-through u_i~1 → u_i; gradient is identity.
        outs: tuple[TensorWrapper, ...] = tuple(inputs)
        if v is None:
            # Match the leaf-Model convention: single output unwrapped to a
            # bare wrapper, multiple outputs as a tuple. Written this way (vs
            # a single ternary) so pyright correctly narrows `outs` past the
            # length check.
            if len(outs) == 1:
                return outs[0]
            return outs
        # Build v_out that simply renames each var~1 → var.
        v_out: ChainRuleDict = {}
        for hist, out in zip(self.input_spec, self.output_spec, strict=True):
            v_out[out] = dict(v.get(hist, {}))
        return (*outs, v_out)


__all__ = ["ConstantExtrapolationPredictor"]
