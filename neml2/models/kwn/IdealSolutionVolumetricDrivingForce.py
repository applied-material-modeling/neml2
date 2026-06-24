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

"""Python-native mirror of the C++ ``IdealSolutionVolumetricDrivingForce`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter, parameters, var_inputs
from ...types import Scalar, log
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("IdealSolutionVolumetricDrivingForce")
class IdealSolutionVolumetricDrivingForce(Model):
    r"""Ideal-solution volumetric Gibbs free energy of precipitation.

    Computes the molar Gibbs free energy of precipitation under the
    ideal-solution approximation,

    $$
    \Delta g = R T \sum_k w_k \ln(c_k / c_k^{\mathrm{eq}}),
    $$

    where the sum runs over the species participating in the precipitate and
    $w_k$ are user-supplied stoichiometric weights (default 1, matching
    the Hu--Cocks "product of all components" convention for compound
    precipitates).
    """

    hit = HitSchema(
        input("temperature", Scalar, "Temperature", attr="_T_var"),
        var_inputs(
            "current_concentrations",
            Scalar,
            "Current matrix concentrations for each species participating in the precipitate",
            attr="_x_vars",
        ),
        output(
            "driving_force",
            Scalar,
            "Molar Gibbs free energy of precipitation",
        ),
        parameter(
            "gas_constant",
            Scalar,
            "Universal gas constant",
            attr="R_g",
        ),
        parameters(
            "equilibrium_concentrations",
            Scalar,
            "Equilibrium matrix concentrations for each species. May be supplied as constants "
            "or as outputs of upstream models (e.g. temperature-dependent interpolations).",
            attr="_x_eq_names",
            allow_nonlinear=True,
        ),
        parameters(
            "weights",
            Scalar,
            "Stoichiometric weights for each species. If a single entry is supplied, the same "
            "weight is used for every species. Default is unit weights for every species, "
            "matching the Hu--Cocks compound-precipitate convention.",
            attr="_w_names",
            default="1",
        ),
    )

    # ``from_hit`` auto-declares the ``gas_constant`` parameter (stored as
    # ``R_g``) and the per-entry ``equilibrium_concentrations`` / ``weights``
    # lists (stashed on ``_x_eq_names`` / ``_w_names``). Annotate so pyright
    # sees the typed wrappers/lists that ``Model.__getattr__`` returns.
    _T_var: str
    _x_vars: list[str]
    _x_eq_names: list[str]
    _w_names: list[str]
    R_g: Scalar

    def __post_init__(self) -> None:
        # Mirror the C++ neml_assert checks: at least one species, matching
        # equilibrium-concentration list length, and weights of length 1 or N.
        n = len(self._x_vars)
        if n == 0:
            raise ValueError(
                f"{type(self).__name__}: requires at least one species; "
                "got an empty current_concentrations list."
            )
        if len(self._x_eq_names) != n:
            raise ValueError(
                f"{type(self).__name__}: number of equilibrium_concentrations "
                f"({len(self._x_eq_names)}) does not match number of "
                f"current_concentrations ({n})."
            )
        nw = len(self._w_names)
        if nw != 1 and nw != n:
            raise ValueError(
                f"{type(self).__name__}: expected 1 or {n} entries in weights, got {nw}."
            )
        # Broadcast a single weight to all species, matching the C++ ctor.
        if nw == 1 and n > 1:
            self._w_names = [self._w_names[0]] * n

    def forward(  # type: ignore[override]
        self,
        T: Scalar,
        *args: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Split positional args: the leading structural inputs (one Scalar per
        # current concentration) followed by the *nl_params pack of
        # mode-3/4-promoted parameters (equilibrium concentrations; weights are
        # ``allow_nonlinear=False``, so they always live as static buffers).
        n = len(self._x_vars)
        xs, nl_params = args[:n], args[n:]
        if len(xs) != n:
            raise ValueError(f"{type(self).__name__} expected {n} inputs, got {len(xs)}")

        R_g = self._get_param("R_g", nl_params, Scalar)
        x_eqs = self._get_param_list("_x_eq_names", nl_params, Scalar)
        ws = self._get_param_list("_w_names", nl_params, Scalar)

        # Forward: Δg = R T Σ wₖ log(cₖ / cₖ_eq). Typed Scalar algebra
        # end-to-end, matching ``IdealSolutionVolumetricDrivingForce::set_value``.
        pref = R_g * T
        sum_log = ws[0] * log(xs[0] / x_eqs[0])
        for w, x, x_eq in zip(ws[1:], xs[1:], x_eqs[1:], strict=True):
            sum_log = sum_log + w * log(x / x_eq)

        dg = pref * sum_log

        if v is None:
            return dg

        # Differential pushforward. Linear coefficients:
        #   ∂Δg/∂T      = R_g · sum_log                  ⇒ action(V) = R_g·sum_log·V
        #   ∂Δg/∂cₖ     = pref · wₖ / cₖ                 ⇒ action(V) = (pref·wₖ/cₖ)·V
        #   ∂Δg/∂cₖ_eq  = -pref · wₖ / cₖ_eq             ⇒ action(V) = (-pref·wₖ/cₖ_eq)·V
        # Accumulate per input name (the +=) so aliased input/parameter names
        # (e.g. the same equilibrium concentration reused across two species)
        # sum exactly as the C++ ``set_value`` does.
        actions: dict[str, ChainRuleAction] = {}

        def _add(name: str, action: ChainRuleAction) -> None:
            if name in actions:
                prev = actions[name]
                actions[name] = lambda V, a=prev, b=action: a(V) + b(V)
            else:
                actions[name] = action

        dg_dT = R_g * sum_log
        _add(self._T_var, lambda V, c=dg_dT: c * V)

        for x_name, w, x in zip(self._x_vars, ws, xs, strict=True):
            coef = pref * w / x
            _add(x_name, lambda V, c=coef: c * V)

        for w, x_eq, p_name in zip(ws, x_eqs, self._x_eq_names, strict=True):
            nlp = self._nl_params.get(p_name)
            if nlp is not None:
                coef = -(pref * w) / x_eq
                _add(nlp.input_name, lambda V, c=coef: c * V)

        return dg, self.apply_chain_rule(v, "driving_force", actions, output=dg)


__all__ = ["IdealSolutionVolumetricDrivingForce"]
