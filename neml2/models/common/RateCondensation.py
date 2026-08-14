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


r"""Condense a rate-to-driving-force model into the pair a coordinate solve needs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...factory import register_neml2_object
from ...schema import HitSchema, dependency, option, output
from ...types import Scalar
from ..model import Model

if TYPE_CHECKING:
    from ..chain_rule import ChainRuleDict


@register_neml2_object("RateCondensation")
class RateCondensation(Model):
    r"""Linearize $f(\dot\gamma)$ at $\dot\gamma = 0$, emitting $b$ and $A$.

    :class:`~neml2.models.common.CoordinateDescentPredictor` solves
    $\varphi(\dot\gamma) + A\dot\gamma = b$, which is the condensed system
    $\dot\gamma = \varphi^{-1}(f(\dot\gamma))$ with $f$ linearized:

    $$f(\dot\gamma) \approx f(0) - A\dot\gamma, \qquad
      b = f(0), \qquad A = -\left.\frac{\partial f}{\partial \dot\gamma}\right|_0.$$

    `model` supplies $f$ -- the **return path**: rate in, driving force out,
    with the flow direction frozen at the trial state. Both $b$ and $A$ come
    from a single :meth:`~neml2.models.model.Model.jvp` of it, which is one
    forward pass with a tangent riding the existing ``forward(v=)`` chain rule
    (see :meth:`~neml2.models.model.Model.jvp`). So this is neither a finite
    difference -- no truncation error, no step size to pick -- nor reverse-mode
    autograd.

    **Why this rather than a closed form.** Writing $A$ out by hand means one
    term per hardening mechanism: $\Delta t\,N:\mathbb{C}:N$ for the elastic
    relaxation, $\Delta t\,h'$ for isotropic hardening, $\Delta t\,N:\partial
    \dot X_i/\partial\dot\gamma$ for each back stress. Every new hardening law
    is then a code change, and the terms are easy to get subtly wrong (the
    kinematic one is $C_i$, not $\tfrac{2}{3}C_i$, because $N:N = \tfrac32$).
    Differentiating the authored path costs nothing per physics: the hardening
    model is a block in `model`, and this leaf never learns any of it. All it
    needs is a first-order chain rule on each leaf in the path, which is the
    framework's own invariant.

    **Dropping the hardening instead does not work.** Freezing it leaves
    $A = \Delta t\,N:\mathbb{C}:N$, whose relative error in $\Delta\gamma$ is
    $h'/(N:\mathbb{C}:N)$ -- under a percent for the usual metal
    parameterizations. But the returned stress is a near-cancellation of the
    trial stress against the plastic relaxation, and it amplifies that error by
    $\sigma^{\rm trial}/\sigma$, which grows with the step. Measured on the
    viscoplasticity scenarios, lagging halves the first-step Newton count below
    $\Delta t \approx 1$ and buys nothing above it; the exact coupling
    converges *at the predictor* at every step size.

    Only the trial-point tangent is taken, so a path that is strongly nonlinear
    over a step is linearized, not solved. That is a warm start, not an
    integrator: a predictor moves the initial guess and never the answer.
    """

    hit = HitSchema(
        dependency(
            "model",
            "get_model",
            "The return path: the rate in, the driving force out, with the flow "
            "direction frozen at the trial state. Its rate input is named by `rate` "
            "and its driving-force output by `driving_force`; every other input is "
            "forwarded unchanged.",
        ),
        option("rate", str, "Which of `model`'s inputs is the rate."),
        option(
            "driving_force",
            str,
            "Which of `model`'s outputs is the driving force.",
        ),
        output(
            "coupling",
            Scalar,
            r"The coupling $A = -\partial f/\partial\dot\gamma$ at zero rate",
            attr="_A",
        ),
        output(
            "trial_driving_force",
            Scalar,
            r"The right-hand side $b = f(0)$: the driving force with the rate at zero",
            attr="_b",
        ),
    )

    _rate: str
    _f: str
    _A: str
    _b: str

    def __init__(self, *, model: Model, rate: str, driving_force: str, **hit_values) -> None:
        # ``**hit_values`` is load-bearing: input_spec is only knowable once the
        # path model is plugged in, and the extension below survives only if
        # schema resolution already ran. See CoordinateDescentPredictor.__init__
        # for the full account of why a named-only signature would silently drop it.
        super().__init__(**hit_values)
        if rate not in model.input_spec:
            raise ValueError(
                f"RateCondensation: rate {rate!r} is not an input of the path model "
                f"{type(model).__name__}; it has {list(model.input_spec)}."
            )
        if driving_force not in model.output_spec:
            raise ValueError(
                f"RateCondensation: driving_force {driving_force!r} is not an output of "
                f"the path model {type(model).__name__}; it has {list(model.output_spec)}."
            )
        # The zero-rate seed below is built at base shape (), which only makes a
        # well-formed wrapper for a scalar. A wrapper with a real base region
        # accepts that shape without complaining -- SR2(torch.zeros(())) is
        # malformed but silent -- and the damage surfaces somewhere inside
        # ``jvp``. ``rate`` is free-text HIT naming any input of the path model,
        # so this is reachable by a wiring mistake, not only by a code change.
        rate_cls = model.input_spec[rate]
        if tuple(rate_cls.BASE_SHAPE) != ():
            raise ValueError(
                f"RateCondensation: rate {rate!r} is {rate_cls.__name__}, whose base "
                f"shape is {tuple(rate_cls.BASE_SHAPE)}; the zero-rate seed is only "
                f"defined for a scalar rate."
            )
        self.model = model
        self._rate = rate
        self._f = driving_force
        # The path's other inputs -- the old state and the forces -- are not ours
        # to invent; surface them so the enclosing graph supplies them.
        self._passthrough = [n for n in model.input_spec if n != rate]
        if not self._passthrough:
            raise ValueError(
                f"RateCondensation: the path model {type(model).__name__} has no inputs "
                f"besides the rate {rate!r}, leaving nothing to take the zero-rate seed's "
                f"dtype and device from. A return path always carries the old state and "
                f"the forces, so this is a wiring mistake rather than a degenerate case."
            )
        self.input_spec = {
            **self.input_spec,
            **{n: model.input_spec[n] for n in self._passthrough},
        }

    def forward(  # type: ignore[override]
        self,
        *args,
        v: ChainRuleDict | None = None,
    ):
        del v  # consumed only by a predictor, which is never differentiated
        vals = dict(zip(self.input_spec, args, strict=True))
        pt = {n: vals[n] for n in self._passthrough}

        # A scalar zero (and unit tangent) with no batch of its own: the batch
        # broadcasts up from the path's other inputs, so this leaf needs no way
        # to ask what the rate's batch shape is. Sound for a scalar rate, which
        # is the one-coordinate condensation. A rate carried on a sub-batch axis
        # (crystal plasticity) would need a seed that is an identity over that
        # axis -- one pass with a K-wide tangent, but not this seed.
        ref = next(iter(pt.values()))
        rate_cls = self.model.input_spec[self._rate]
        zero = rate_cls(torch.zeros((), dtype=ref.dtype, device=ref.device))  # type: ignore[call-arg]
        one = rate_cls(torch.ones((), dtype=ref.dtype, device=ref.device))  # type: ignore[call-arg]

        out, dout = self.model.jvp({self._rate: zero, **pt}, {self._rate: one})
        return -dout[self._f], out[self._f]


__all__ = ["RateCondensation"]
