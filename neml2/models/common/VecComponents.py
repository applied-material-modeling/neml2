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

"""Python-native mirror of C++ ``common/VecComponents.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, option
from ...types import (
    Scalar,
    Vec,
    vec_component,
)
from .._hit import _opt_list_str


def _read_list_str(node, name):  # noqa: ANN001, ANN202
    return list(node.param_list_str(name))


@register_neml2_object("VecComponents")
class VecComponents(Model):
    """Decompose a Vec into its three Scalar components.

    Wraps the C++ ``Vec::operator()(int)`` slot access as a registered Model
    so component extraction can participate in ``ComposedModel`` assemblies.
    The three output names are user-supplied via the ``to = 's1 s2 s3'``
    option (resolved positionally as components 0, 1, 2 of the input ``Vec``).

    Linear leaf: the forward is ``to[i] = from(i)`` and the D-062 pushforward
    is ``action_i(V) = V(i)`` for any incoming ``Vec`` tangent $V$.
    """

    hit = HitSchema(
        input("from", Vec, "The Vec to decompose", attr="_from_var"),
        option(
            "to",
            list,
            "Names of the three Scalar component outputs (in order: 0, 1, 2).",
            reader=_read_list_str,
            optional_reader=_opt_list_str,
            attr="_to_vars",
        ),
    )

    _from_var: str
    _to_vars: list[str]

    def __init__(self, **hit_values) -> None:
        super().__init__(**hit_values)
        if len(self._to_vars) != 3:
            raise ValueError(
                f"VecComponents requires exactly 3 output names; got {len(self._to_vars)}."
            )
        # Static spec already carries the renamed ``_from_var`` (from
        # _store_schema_values' input handling); extend output_spec to hold
        # one Scalar entry per user-supplied name in declaration order.
        self.output_spec = {name: Scalar for name in self._to_vars}

    def forward(  # type: ignore[override]
        self,
        inp: Vec,
        *nl_params: object,
        v: ChainRuleDict | None = None,
    ):
        # Forward: extract components 0, 1, 2 via the typed wrapper helper.
        # No .data, no torch.* -- vec_component preserves sub-batch metadata.
        s0 = vec_component(inp, 0)
        s1 = vec_component(inp, 1)
        s2 = vec_component(inp, 2)
        if v is None:
            return s0, s1, s2

        # Linear leaf: d(to[i])/d(from) is the i-th unit Vec; the action on
        # an incoming Vec tangent V is just its i-th component. No materialised
        # Jacobian; the chain rule pushforward is the same per-axis pick as
        # the forward.
        def make_action(i: int):
            return lambda V: vec_component(V, i)

        v0 = self.apply_chain_rule(v, self._to_vars[0], {self._from_var: make_action(0)}, output=s0)
        v1 = self.apply_chain_rule(v, self._to_vars[1], {self._from_var: make_action(1)}, output=s1)
        v2 = self.apply_chain_rule(v, self._to_vars[2], {self._from_var: make_action(2)}, output=s2)
        # Merge per-output sensitivity dicts; the three output keys are
        # distinct (user-supplied names) so a plain dict union is safe.
        return s0, s1, s2, {**v0, **v1, **v2}


__all__ = ["VecComponents"]
