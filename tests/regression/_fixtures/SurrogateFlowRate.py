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

"""Machine-learning surrogate flow rate, demonstrating ``request_AD``.

This is the v3 replacement for the old ``TorchScriptFlowRate`` fixture. In v2 the
authoring surface was C++, so deploying a Python model into NEML2 meant exporting
it to a TorchScript ``.pt`` and loading it through bespoke plumbing, and its
Jacobian had to be taken with ``torch.func.jvp`` (forward-mode, which does *not*
lower through AOTI). In v3 Python *is* the authoring surface, so the surrogate is
just an ordinary :class:`torch.nn.Module` and its first-order chain rule is
supplied by :meth:`~neml2.models.model.Model.request_AD` -- reverse-mode autodiff
that the framework lowers through every route.

This is the canonical pattern for integrating a trained machine-learning model
into NEML2: drop the network in as the forward operator, declare ``request_AD``,
and never hand-write (or forward-mode-autodiff) its Jacobian. Here the "network"
is a closed-form Perzyna viscoplastic law with grain-size / stoichiometry /
Arrhenius sensitivities -- reproducing the exact numerics of the retired
``surrogate.pt`` so the regression gold is unchanged -- but a genuine MLP would be
wired in identically.

Test-fixture only; self-registers via ``@register_neml2_object``.
"""

from __future__ import annotations

import torch

from neml2.factory import register_neml2_object
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import Scalar
from neml2.types.functions import exp_ad


class _PerzynaSurrogate(torch.nn.Module):
    """Stand-in for a trained flow-rate network (raw ``torch`` in / out).

    ``ep_dot = exp(-Q/(R T)) * (exp(-G/G0) + exp(-C/C0)) * <(|s - sy|/eta)^n
    Heaviside(s - sy)>`` with the grain size ``G`` and stoichiometry ``C`` held at
    the fixed values the original C++ helper used. The saved-output ``exp`` is
    routed through :func:`neml2.types.functions.exp_ad` so the reverse-mode
    Jacobian that ``request_AD`` builds also lowers through AOTI (the
    saved-output #187907 boundary); ``pow`` / ``abs`` / ``sign`` save their inputs
    and need no special handling.
    """

    # Narrow the registered-buffer attribute types (nn.Module.__getattr__ is
    # otherwise ``Tensor | Module`` to pyright).
    sy: torch.Tensor
    eta: torch.Tensor
    n: torch.Tensor
    G0: torch.Tensor
    C0: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        consts = dict(sy=1000.0, eta=10.0, n=3.0, G0=0.003, C0=0.004, Q=50000.0, R=8.3145)
        for name, value in consts.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))

    def forward(self, s: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        # Grain size / stoichiometry constants matching the C++ Scalar::full fills.
        G = torch.full_like(s, 0.1)
        C = torch.full_like(s, 0.2)
        f = s - self.sy
        Hf = (torch.sign(f) + 1.0) / 2.0  # Heaviside: 1 where s > sy, else 0
        rep = torch.pow(torch.abs(f) / self.eta, self.n) * Hf
        rG = exp_ad(-G / self.G0)
        rC = exp_ad(-C / self.C0)
        aT = exp_ad(-self.Q / self.R / T)
        return aT * (rG + rC) * rep


@register_neml2_object("SurrogateFlowRate")
class SurrogateFlowRate(Model):
    """Flow-rate model backed by a Python ``torch`` surrogate, differentiated by
    ``request_AD``.

    Schema: 2 Scalar inputs (von Mises stress, temperature), 1 Scalar output
    (equivalent plastic strain rate). The leaf writes only the value forward;
    ``request_AD`` (declared in :meth:`__post_init__`) supplies the first-order
    chain rule ``d(ep_dot)/d(s)`` and ``d(ep_dot)/d(T)`` that the implicit
    radial-return solve needs.
    """

    hit = HitSchema(
        input("von_mises_stress", Scalar, "The von Mises stress", attr="_s_var"),
        input("temperature", Scalar, "The temperature", attr="_T_var"),
        output(
            "equivalent_plastic_strain_rate",
            Scalar,
            "The equivalent plastic strain rate",
            attr="_ep_dot_var",
        ),
    )

    _s_var: str
    _T_var: str
    _ep_dot_var: str
    # The surrogate is an ``nn.Module`` attribute, not a NEML2 parameter; narrow
    # its type so pyright sees ``self._surrogate(...)`` as callable.
    _surrogate: _PerzynaSurrogate

    def __post_init__(self) -> None:
        self._surrogate = _PerzynaSurrogate()
        # Auto-derive the first-order chain rule for both structural inputs.
        self.request_AD()

    def forward(self, s: Scalar, T: Scalar, *nl_params: Scalar):  # type: ignore[override]
        # ML-surrogate raw-tensor boundary: the network consumes/produces raw
        # ``torch.Tensor``; unwrap to feed it and re-wrap the result. request_AD
        # differentiates straight through this boundary (the grad-tracking input
        # leaf it swaps in is exactly ``s.data`` / ``T.data``).
        out = self._surrogate(s.data, T.data)  # noqa: data-ok ML-surrogate boundary
        return Scalar(out)


__all__ = ["SurrogateFlowRate"]
