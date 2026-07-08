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

"""Python-native mirror of the C++ ``ProjectedDiffusivitySum`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, output, parameters, var_inputs
from ...types import Scalar, pow
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("ProjectedDiffusivitySum")
class ProjectedDiffusivitySum(Model):
    """Compute the projected diffusivity sum for SFFK and nucleation."""

    hit = HitSchema(
        var_inputs(
            "far_field_concentrations",
            Scalar,
            "Far-field concentrations",
            attr="_x_inf_vars",
        ),
        output(
            "projected_diffusivity_sum",
            Scalar,
            "Projected diffusivity sum",
        ),
        parameters(
            "concentration_differences",
            Scalar,
            "Concentration differences for each species",
            attr="_dx_names",
            allow_promotion=True,
        ),
        parameters(
            "diffusivities",
            Scalar,
            "Species diffusivities",
            attr="_D_names",
            allow_promotion=True,
        ),
    )

    # ``from_hit`` auto-declares the per-entry ``concentration_differences`` /
    # ``diffusivities`` parameter lists (stashed on ``_dx_names`` / ``_D_names``)
    # and the ``far_field_concentrations`` input list (stashed on
    # ``_x_inf_vars``). Annotate so pyright sees the resolved name lists.
    _x_inf_vars: list[str]
    _dx_names: list[str]
    _D_names: list[str]

    def __post_init__(self) -> None:
        # Mirror the C++ neml_assert checks: at least one species, matching
        # diffusivity / far-field-concentration list lengths.
        n = len(self._dx_names)
        if n == 0:
            raise ValueError(
                f"{type(self).__name__} requires at least one species; got an empty "
                "concentration_differences list."
            )
        if len(self._D_names) != n:
            raise ValueError(
                f"{type(self).__name__}: number of diffusivities ({len(self._D_names)}) does not "
                f"match number of concentration differences ({n})."
            )
        if len(self._x_inf_vars) != n:
            raise ValueError(
                f"{type(self).__name__}: number of far-field concentrations "
                f"({len(self._x_inf_vars)}) does not match number of concentration differences "
                f"({n})."
            )

    def forward(  # type: ignore[override]
        self,
        *args: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Split positional args: the leading structural inputs (one Scalar per
        # far-field concentration) followed by the *promoted_params pack of
        # mode-3/4-promoted concentration_differences / diffusivities entries.
        n = len(self._x_inf_vars)
        x_infs, promoted_params = args[:n], args[n:]
        if len(x_infs) != n:
            raise ValueError(f"{type(self).__name__} expected {n} inputs, got {len(x_infs)}")

        dxs = self._get_param_list("_dx_names", promoted_params, Scalar)
        Ds = self._get_param_list("_D_names", promoted_params, Scalar)

        # Forward: s = sum_k dx_k^2 / (D_k * x_inf_k). Typed Scalar algebra
        # end-to-end, matching ``ProjectedDiffusivitySum::set_value``.
        s = pow(dxs[0], 2.0) / (Ds[0] * x_infs[0])
        for dx, D, xinf in zip(dxs[1:], Ds[1:], x_infs[1:], strict=True):
            s = s + pow(dx, 2.0) / (D * xinf)

        if v is None:
            return s

        # Differential pushforward. Per-species partial derivatives are
        # all closed-form linear coefficients:
        #   d s / d dx_k    = 2 * dx_k / (D_k * x_inf_k)
        #   d s / d x_inf_k = - dx_k^2 / (D_k * x_inf_k^2)
        #   d s / d D_k     = - dx_k^2 / (D_k^2 * x_inf_k)
        # Accumulate per input name (the +=) so aliased input/parameter names
        # (e.g. the same far-field concentration reused across two species)
        # sum exactly as the C++ ``set_value`` does.
        actions: dict[str, ChainRuleAction] = {}

        def _add(name: str, action: ChainRuleAction) -> None:
            if name in actions:
                prev = actions[name]
                actions[name] = lambda V, a=prev, b=action: a(V) + b(V)
            else:
                actions[name] = action

        for x_name, dx, D, xinf in zip(self._x_inf_vars, dxs, Ds, x_infs, strict=True):
            coef = -pow(dx, 2.0) / (D * xinf * xinf)
            _add(x_name, lambda V, c=coef: c * V)

        for dx, D, xinf, p_name in zip(dxs, Ds, x_infs, self._dx_names, strict=True):
            pparam = self._promoted_params.get(p_name)
            if pparam is not None:
                coef = (2.0 * dx) / (D * xinf)
                _add(pparam.input_name, lambda V, c=coef: c * V)

        for dx, D, xinf, p_name in zip(dxs, Ds, x_infs, self._D_names, strict=True):
            pparam = self._promoted_params.get(p_name)
            if pparam is not None:
                coef = -pow(dx, 2.0) / (D * D * xinf)
                _add(pparam.input_name, lambda V, c=coef: c * V)

        return s, self.apply_chain_rule(v, "projected_diffusivity_sum", actions, output=s)


__all__ = ["ProjectedDiffusivitySum"]
