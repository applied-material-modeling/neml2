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

"""Python-native mirror of the C++ ``ChemicalGibbsFreeEnergyDifference`` model."""

from __future__ import annotations

from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, output, parameters, var_inputs
from ...types import Scalar


@register_native("ChemicalGibbsFreeEnergyDifference")
class ChemicalGibbsFreeEnergyDifference(Model):
    r"""Chemical Gibbs free energy difference for a set of species.

    Computes :math:`\Delta g = \sum_i \Delta x_i \, (\mu_i - \mu_i^{\mathrm{eq}})`
    where :math:`\Delta x_i` is the concentration difference for species
    :math:`i`, :math:`\mu_i` is the chemical potential in the matrix, and
    :math:`\mu_i^{\mathrm{eq}}` is the equilibrium chemical potential.
    """

    hit = HitSchema(
        var_inputs(
            "concentration_differences",
            Scalar,
            "Concentration differences for each species",
            attr="_dx_vars",
        ),
        output(
            "chemical_gibbs_free_energy",
            Scalar,
            "Chemical Gibbs free energy difference",
        ),
        parameters(
            "chemical_potentials",
            Scalar,
            "Chemical potentials in the matrix",
            attr="_mu_names",
            allow_nonlinear=True,
        ),
        parameters(
            "equilibrium_potentials",
            Scalar,
            "Equilibrium chemical potentials",
            attr="_mu_eq_names",
            allow_nonlinear=True,
        ),
    )

    # ``parameters(...)`` stores the per-entry parameter-name lists on these
    # attributes; ``_get_param_list`` later reads them back via ``_get_param``.
    _dx_vars: list[str]
    _mu_names: list[str]
    _mu_eq_names: list[str]

    def __post_init__(self) -> None:
        # Mirror the C++ neml_assert checks: matching list lengths and at least
        # one species. The schema's ``parameters(...)`` fields can be empty
        # lists if the user passes nothing, and a length mismatch would only
        # surface as a confusing IndexError inside ``forward``.
        n = len(self._dx_vars)
        if n == 0:
            raise ValueError(
                f"{type(self).__name__}: requires at least one species; "
                "got an empty concentration_differences list."
            )
        if len(self._mu_names) != n:
            raise ValueError(
                f"{type(self).__name__}: number of chemical_potentials "
                f"({len(self._mu_names)}) does not match number of "
                f"concentration_differences ({n})."
            )
        if len(self._mu_eq_names) != n:
            raise ValueError(
                f"{type(self).__name__}: number of equilibrium_potentials "
                f"({len(self._mu_eq_names)}) does not match number of "
                f"concentration_differences ({n})."
            )

    def forward(  # type: ignore[override]
        self,
        *args: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Split positional args: leading structural inputs (one Scalar per
        # species) followed by the *nl_params pack of mode-3/4-promoted
        # potentials (if any).
        n = len(self._dx_vars)
        dxs, nl_params = args[:n], args[n:]
        if len(dxs) != n:
            raise ValueError(f"{type(self).__name__} expected {n} inputs, got {len(dxs)}")

        mus = self._get_param_list("_mu_names", nl_params, Scalar)
        mu_eqs = self._get_param_list("_mu_eq_names", nl_params, Scalar)

        # Forward: Δg = Σ Δxᵢ (μᵢ − μᵢ_eq)
        coefs = [mu - mu_eq for mu, mu_eq in zip(mus, mu_eqs, strict=True)]
        dg = coefs[0] * dxs[0]
        for c, dx in zip(coefs[1:], dxs[1:], strict=True):
            dg = dg + c * dx

        if v is None:
            return dg

        # Differential pushforward. Linear in every variable:
        #   ∂Δg/∂Δxᵢ      = (μᵢ − μᵢ_eq)   ⇒ action(V) = coefᵢ · V
        #   ∂Δg/∂μᵢ       = +Δxᵢ            ⇒ action(V) = +Δxᵢ · V
        #   ∂Δg/∂μᵢ_eq    = −Δxᵢ            ⇒ action(V) = −Δxᵢ · V
        # Accumulate per input name (the +=) so that aliased parameter names
        # (e.g. the same μ reused across two species, or a μ that happens to
        # share an input name with another promoted parameter) sum exactly as
        # the C++ ``set_value`` does.
        actions: dict[str, ChainRuleAction] = {}

        def _add(name: str, action: ChainRuleAction) -> None:
            if name in actions:
                prev = actions[name]
                actions[name] = lambda V, a=prev, b=action: a(V) + b(V)
            else:
                actions[name] = action

        for dx_name, c in zip(self._dx_vars, coefs, strict=True):
            _add(dx_name, lambda V, c=c: c * V)

        for i, p_name in enumerate(self._mu_names):
            nlp = self._nl_params.get(p_name)
            if nlp is not None:
                _add(nlp.input_name, lambda V, dx=dxs[i]: dx * V)

        for i, p_name in enumerate(self._mu_eq_names):
            nlp = self._nl_params.get(p_name)
            if nlp is not None:
                _add(nlp.input_name, lambda V, dx=dxs[i]: -(dx * V))

        return dg, self.apply_chain_rule(v, "chemical_gibbs_free_energy", actions, output=dg)


__all__ = ["ChemicalGibbsFreeEnergyDifference"]
