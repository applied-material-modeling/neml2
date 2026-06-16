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

"""Python-native mirror of the C++ ``YieldFunction`` model."""

from __future__ import annotations

import math

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ...model import Model


@register_neml2_object("YieldFunction")
class YieldFunction(Model):
    r"""Classical macroscale plasticity yield function,
    $f = \bar{\sigma} - \sigma_y - h$, where $\bar{\sigma}$ is the
    effective stress, $\sigma_y$ is the yield stress, and $h$ is the
    isotropic hardening (optional: omit the ``isotropic_hardening`` HIT option
    to drop the $- h$ term entirely).
    """

    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        input("effective_stress", Scalar, "Effective stress", attr="_s_name"),
        # ``default=None`` pops ``isotropic_hardening`` from ``input_spec`` when
        # HIT is silent, mirroring the C++ ``add_optional_input`` knob. The
        # resolved name (or ``None``) lands on the instance as ``_h_name``.
        input(
            "isotropic_hardening",
            Scalar,
            "Isotropic hardening",
            default=None,
            attr="_h_name",
        ),
        output("yield_function", Scalar, "Yield function"),
        parameter("yield_stress", Scalar, "Yield stress", attr="sy", allow_nonlinear=True),
    )

    # ``from_hit`` auto-declares the ``yield_stress`` parameter (stored as
    # ``sy``) — no __init__ needed.
    sy: Scalar
    _s_name: str
    _h_name: str | None

    _SQRT_2_3 = math.sqrt(2.0 / 3.0)

    def forward(  # type: ignore[override]
        self,
        *inputs: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # Inputs arrive positionally in ``input_spec`` declaration order. When
        # HIT omits ``isotropic_hardening`` it's been popped from
        # ``input_spec`` (default=None), so the present subset is exactly
        # the present positional args — pair them with the surviving names.
        names = list(self.input_spec)
        n_structural = 1 + (1 if self._h_name is not None else 0)
        if len(inputs) < n_structural:
            raise AssertionError(
                f"YieldFunction.forward: got {len(inputs)} inputs, expected at least "
                f"{n_structural} structural inputs"
            )
        structural = dict(zip(names[:n_structural], inputs[:n_structural], strict=True))
        nl_params = inputs[n_structural:]

        effective_stress = structural[self._s_name]
        sy = self._get_param("sy", nl_params, Scalar)
        c = self._SQRT_2_3
        f = c * (effective_stress - sy)
        if self._h_name is not None:
            f = f - c * structural[self._h_name]
        if v is None:
            return f

        actions_1 = {
            self._s_name: lambda V, c_=c: c_ * V,
        }
        if self._h_name is not None:
            actions_1[self._h_name] = lambda V, c_=c: -c_ * V
        if "sy" in self._nl_params:
            sy_input_name = self._nl_params["sy"].input_name
            actions_1[sy_input_name] = lambda V, c_=c: -c_ * V
        # Linear in every input ⇒ second derivatives are zero (no actions_2).
        return f, *self.propagate_tangents(v, "yield_function", actions_1, output=f, v2=v2, vh=vh)
