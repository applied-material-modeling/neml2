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

"""Python-native mirror of the C++ ``EffectiveVolume`` model."""

from __future__ import annotations

from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, output, parameter, parameters, var_inputs
from ...types import Scalar


@register_neml2_object("EffectiveVolume")
class EffectiveVolume(Model):
    r"""Total volume of a control-mass composite during a reaction.

    .. math::

        V = \frac{M}{1 - \phi_o} \sum_i \frac{\omega_i}{\rho_i}

    where :math:`\omega_i` and :math:`\rho_i` are the mass fraction and density
    of each component, :math:`\phi_o` is the open volume fraction (accounting
    for leakage out of the control mass), and :math:`M` is the reference mass
    of the composite. The open-volume-fraction input is optional - when HIT is
    silent the prefactor reduces to :math:`M`.
    """

    hit = HitSchema(
        # Optional open volume fraction; default=None pops it from input_spec
        # when HIT is silent (mirrors the C++ ``add_input`` + ``defined`` knob).
        input(
            "open_volume_fraction",
            Scalar,
            "Open volume fraction accounting for leakage",
            default=None,
            attr="_phio_name",
        ),
        var_inputs(
            "mass_fractions",
            Scalar,
            "Mass fractions of the components in the composite",
            attr="_w_names",
        ),
        output("composite_volume", Scalar, "Volume of the composite"),
        parameter(
            "reference_mass",
            Scalar,
            "Reference mass of the composite",
            attr="M",
            allow_nonlinear=True,
        ),
        parameters(
            "densities",
            Scalar,
            "Densities of the components in the composite",
            attr="_rho_names",
        ),
    )

    # ``from_hit`` auto-declares the ``reference_mass`` (stored as ``M``) and
    # each ``densities`` entry (stored as ``densities_0``, ...). Annotations let
    # pyright see the typed wrapper that ``Model.__getattr__`` returns.
    M: Scalar
    _phio_name: str | None
    _w_names: list[str]
    _rho_names: list[str]

    def __post_init__(self) -> None:
        if len(self._rho_names) != len(self._w_names):
            raise ValueError(
                f"{type(self).__name__}: number of mass fractions "
                f"({len(self._w_names)}) does not match number of densities "
                f"({len(self._rho_names)})."
            )

    def forward(  # type: ignore[override]
        self,
        *args,
        v: ChainRuleDict | None = None,
        **_,
    ):
        # Inputs arrive positionally in ``input_spec`` declaration order, then
        # the *nl_params pack. The optional ``open_volume_fraction`` entry is
        # popped from ``input_spec`` when HIT didn't name it — pair the present
        # subset with the surviving names.
        names = list(self.input_spec)
        n_in = len(names)
        inputs, nl_params = args[:n_in], args[n_in:]
        if len(inputs) != n_in:
            raise AssertionError(
                f"EffectiveVolume.forward: got {len(inputs)} inputs, expected {n_in}"
            )
        bound = dict(zip(names, inputs, strict=True))

        ws = [bound[n] for n in self._w_names]
        phio_name = self._phio_name
        phio = bound[phio_name] if phio_name is not None and phio_name in bound else None

        M = self._get_param("M", nl_params, Scalar)
        rhos = self._get_param_list("_rho_names", nl_params, Scalar)

        # sum = Σ_i (w_i / rho_i)
        terms = [w / rho for w, rho in zip(ws, rhos, strict=True)]
        sum_term = terms[0]
        for t in terms[1:]:
            sum_term = sum_term + t

        # coef = M / (1 - phio) when phio is present, else M.
        if phio is None:
            coef = M
        else:
            coef = M / (1.0 - phio)

        V = coef * sum_term
        if v is None:
            return V

        # Differential pushforward.
        # ∂V/∂w_i = coef / rho_i  (linear in each w_i)
        # ∂V/∂phio = M * sum / (1 - phio)^2 = coef * sum / (1 - phio)
        actions: dict[str, ChainRuleAction] = {}
        for w_name, rho in zip(self._w_names, rhos, strict=True):
            c_i = coef / rho
            actions[w_name] = lambda V_, c=c_i: c * V_
        if phio is not None and phio_name is not None and phio_name in bound:
            c_phio = coef * sum_term / (1.0 - phio)
            actions[phio_name] = lambda V_, c=c_phio: c * V_

        return V, self.apply_chain_rule(v, "composite_volume", actions, output=V)


__all__ = ["EffectiveVolume"]
