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

"""Python-native mirror of the C++ ``CurrentConcentration`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, output, parameter, parameters, var_inputs
from ...types import Scalar
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("CurrentConcentration")
class CurrentConcentration(Model):
    r"""Compute the current matrix concentration from precipitate volume fractions.

    For $N$ precipitate species with volume fractions $f_i$ and
    (constant) precipitate concentrations $x_i^p$, the matrix
    concentration is

    $$
    x = \frac{x_0 - \sum_i f_i x_i^p}{1 - \sum_i f_i}
    $$

    where $x_0$ is the initial concentration in solution.
    """

    hit = HitSchema(
        var_inputs(
            "precipitate_volume_fractions",
            Scalar,
            "Precipitate volume fraction variables",
            attr="_f_vars",
        ),
        output(
            "current_concentration",
            Scalar,
            "Current concentration in solution",
        ),
        parameter(
            "initial_concentration",
            Scalar,
            "Initial concentration in solution",
            attr="x0",
            allow_nonlinear=True,
        ),
        parameters(
            "precipitate_concentrations",
            Scalar,
            "Precipitate concentrations",
            attr="_xp_names",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``initial_concentration`` parameter
    # (stored as ``x0``) and the list of ``precipitate_concentrations``
    # entries (per-entry names stashed on ``_xp_names``). Annotate so pyright
    # sees the typed wrappers/lists that ``Model.__getattr__`` returns.
    _f_vars: list[str]
    _xp_names: list[str]
    x0: Scalar

    def __post_init__(self) -> None:
        # Mirror the C++ neml_assert checks: at least one precipitate, and
        # matching list lengths for fractions vs concentrations.
        n = len(self._f_vars)
        if n == 0:
            raise ValueError(
                f"{type(self).__name__}: requires at least one precipitate; "
                "got an empty precipitate_volume_fractions list."
            )
        if len(self._xp_names) != n:
            raise ValueError(
                f"{type(self).__name__}: number of precipitate_concentrations "
                f"({len(self._xp_names)}) does not match number of "
                f"precipitate_volume_fractions ({n})."
            )

    def forward(  # type: ignore[override]
        self,
        *args: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Split positional args: the leading structural inputs (one Scalar per
        # precipitate volume fraction) followed by the *nl_params pack of
        # mode-3/4-promoted parameters (initial concentration and/or any of the
        # per-species precipitate concentrations).
        n = len(self._f_vars)
        fs, nl_params = args[:n], args[n:]
        if len(fs) != n:
            raise ValueError(f"{type(self).__name__} expected {n} inputs, got {len(fs)}")

        x0 = self._get_param("x0", nl_params, Scalar)
        xps = self._get_param_list("_xp_names", nl_params, Scalar)

        # Forward: x = (x0 - Σ fᵢ xᵢ^p) / (1 - Σ fᵢ). Typed Scalar algebra
        # end-to-end, matching ``CurrentConcentration::set_value``.
        sum_f = fs[0]
        sum_fx = fs[0] * xps[0]
        for f, xp in zip(fs[1:], xps[1:], strict=True):
            sum_f = sum_f + f
            sum_fx = sum_fx + f * xp

        denom = 1.0 - sum_f
        numer = x0 - sum_fx
        x = numer / denom

        if v is None:
            return x

        # Differential pushforward. Linear coefficients:
        #   ∂x/∂fᵢ    = (numer - denom · xᵢ^p) / denom²
        #              = (x - xᵢ^p) / denom
        #   ∂x/∂x0    = +1 / denom
        #   ∂x/∂xᵢ^p  = -fᵢ / denom
        # The parameter actions only fire when the parameter has been promoted
        # to a nonlinear input via the HIT ``[Models]`` cross-ref form. Use
        # the post-divide identity ``(numer - denom · xp) / denom² = (x - xp)
        # / denom`` to avoid recomputing the squared denominator.
        inv_denom = 1.0 / denom
        actions: dict[str, ChainRuleAction] = {}

        def _add(name: str, action: ChainRuleAction) -> None:
            # Aliased input or parameter names (same f reused across slots, or a
            # μ that happens to share an input name with a promoted parameter)
            # accumulate via +, matching the C++ ``set_value`` summation.
            if name in actions:
                prev = actions[name]
                actions[name] = lambda V, a=prev, b=action: a(V) + b(V)
            else:
                actions[name] = action

        for f_name, xp in zip(self._f_vars, xps, strict=True):
            coef = (x - xp) * inv_denom
            _add(f_name, lambda V, c=coef: c * V)

        x0_nlp = self._nl_params.get("x0")
        if x0_nlp is not None:
            _add(x0_nlp.input_name, lambda V, c=inv_denom: c * V)

        for f, p_name in zip(fs, self._xp_names, strict=True):
            nlp = self._nl_params.get(p_name)
            if nlp is not None:
                coef = -f * inv_denom
                _add(nlp.input_name, lambda V, c=coef: c * V)

        return x, self.apply_chain_rule(v, "current_concentration", actions, output=x)


__all__ = ["CurrentConcentration"]
