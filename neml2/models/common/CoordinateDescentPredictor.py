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

r"""Warm start by coordinate descent on a condensed rate system.

Many implicit constitutive updates condense onto a vector of rates
$\dot{\gamma} \in \mathbb{R}^m$ as

$$\varphi(\dot{\gamma}) + A\dot{\gamma} = b,$$

where $\varphi$ acts componentwise (the constitutive law, inverted) and
$A \succeq 0$ couples the components through elasticity. Crystal plasticity is
the motivating case: $A = \Delta t\,M^{\mathsf T}\mathbb{C}M$ with $M$ the Schmid
tensors, $b$ the trial resolved shear.

That structure is the gradient of a strictly convex potential -- $\varphi$ is
monotone, so its antiderivative is strictly convex, and $A$ is positive
semi-definite -- which makes **coordinate descent globally convergent with no
globalization at all**, each coordinate step an exact scalar solve rather than a
linearization. Newton, by contrast, crawls from a near-zero start because
$\varphi'$ is unbounded at the origin for a rate-sensitive law.

The physics hook is one model: **the explicit rate law itself**, driving force in
and rate out, i.e. $\varphi^{-1}$. Every rate-dependent physics already has one
(``PowerLawSlipRule`` for crystal plasticity, a Perzyna flow rate for
viscoplasticity), so specializing this predictor to a new physics means supplying
a coupling matrix and pointing at a law that already exists.

Measured on a single-crystal scenario, warm-starting the *rate* (non-inverted)
formulation this way takes the first time step from 16 Newton iterations to 3,
against an oracle ceiling of 4, and widens the convergence basin fourfold. See
``studies/nlprecond/theory/cd_predictor.pdf`` for the derivation.

This model outputs the **rate**. Converting that into whatever unknowns the
implicit system actually carries -- for crystal plasticity, the elastic strain
and the slip hardening -- is left to ordinary model composition downstream, so
no physics enters this class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...factory import register_neml2_object
from ...schema import HitSchema, dependency, input, option, output
from ...types import Scalar, lt, stack, where
from ..model import Model

if TYPE_CHECKING:
    from ..chain_rule import ChainRuleDict


@register_neml2_object("CoordinateDescentPredictor")
class CoordinateDescentPredictor(Model):
    r"""Coordinate descent on $\varphi(\dot{\gamma}) + A\dot{\gamma} = b$ as a warm start.

    Gauss-Seidel over the components, each coordinate solved by bisection in the
    driving-force variable. See the module docstring for the structure and why
    coordinate descent suits it.

    Both $A$ and $b$ are plain :class:`~neml2.types.Scalar`\ s carried on
    sub-batch axes -- $b$ over ``(m,)`` and $A$ over ``(m, m)``. A matrix of
    scalars is exactly what the sub-batch machinery already represents, so this
    needs no dynamic-base tensor at the model boundary.
    """

    hit = HitSchema(
        dependency(
            "rate_law",
            "get_model",
            "Explicit rate law, driving force in and rate out -- the inverse of the "
            "separable nonlinearity. Its driving-force input is named by "
            "`driving_force_input`; every other input is forwarded unchanged.",
        ),
        option(
            "driving_force_input",
            str,
            "Which of `rate_law`'s inputs is the driving force.",
        ),
        input(
            "coupling",
            Scalar,
            "The coupling matrix $A$, sub-batched over (m, m), symmetric positive "
            "semi-definite. Its diagonal must be non-negative -- that is what "
            "brackets each coordinate solve.",
            attr="_A",
        ),
        input(
            "trial_driving_force",
            Scalar,
            "The right-hand side $b$, sub-batched over (m,): the driving force that "
            "would act with every rate at zero (for crystal plasticity, the trial "
            "resolved shear).",
            attr="_b",
        ),
        output("rate", Scalar, "The predicted rate, sub-batched over (m,)", attr="_g"),
        option(
            "sweeps",
            int,
            "Gauss-Seidel sweeps. Measured: 16 suffices; more does not help once the "
            "coupling linearization dominates the error.",
            default="16",
        ),
        option(
            "bisections",
            int,
            "Bisections of the bracket in each coordinate solve, before the Newton "
            "polish. Measured: unchanged from 6 upward and broken at 4, so the "
            "default carries roughly threefold margin.",
            default="16",
        ),
        option(
            "polish",
            int,
            "Newton steps after bisection, using the rate law's own tangent. Three "
            "recovers full relative accuracy from a coarse bracket.",
            default="3",
        ),
    )

    _A: str
    _b: str
    _g: str

    def __init__(
        self,
        *,
        rate_law: Model,
        driving_force_input: str,
        sweeps: int = 16,
        bisections: int = 16,
        polish: int = 3,
    ) -> None:
        super().__init__()
        if driving_force_input not in rate_law.input_spec:
            raise ValueError(
                f"CoordinateDescentPredictor: driving_force_input {driving_force_input!r} "
                f"is not an input of the rate law {type(rate_law).__name__}; it has "
                f"{list(rate_law.input_spec)}."
            )
        if len(rate_law.output_spec) != 1:
            raise ValueError(
                f"CoordinateDescentPredictor: the rate law must have exactly one output "
                f"(the rate); {type(rate_law).__name__} has {list(rate_law.output_spec)}."
            )
        self.rate_law = rate_law
        self.driving_force_input = driving_force_input
        self.sweeps = int(sweeps)
        self.bisections = int(bisections)
        self.polish = int(polish)
        # The rate law's other inputs (e.g. the slip strengths) are not ours to
        # invent -- surface them so the caller supplies them, exactly as
        # ImplicitUpdate surfaces a predictor's extra inputs.
        self._passthrough = [n for n in rate_law.input_spec if n != driving_force_input]
        for name in self._passthrough:
            self.input_spec.setdefault(name, rate_law.input_spec[name])

    # ------------------------------------------------------------------

    def _phi_inv(self, w: Scalar, pt: dict[str, Scalar]) -> Scalar:
        """The rate at driving force *w*: one call of the explicit rate law."""
        out = self.rate_law.call_by_name({self.driving_force_input: w, **pt})
        return next(iter(out.values()))  # type: ignore[return-value]

    def _coord_solve(self, aii: Scalar, c: Scalar, pt: dict[str, Scalar]) -> Scalar:
        r"""Root of $w + A_{ii}\varphi^{-1}(w) = c$, returned as the rate.

        Solved in the driving force $w = \varphi(\dot\gamma)$ rather than in the
        rate: the map is vertical at the origin in the rate -- that is the whole
        pathology -- but has unit slope there in $w$.

        The root is always bracketed by $[0, c]$, free and with no search: at
        $w = 0$ the residual is $-c$, and at $w = c$ it is $A_{ii}\varphi^{-1}(c)$,
        which carries the sign of $c$ because $A_{ii} \ge 0$ and $\varphi^{-1}$
        preserves sign. The upper end is the $A_{ii} = 0$ solution, i.e. the plain
        explicit rate law -- so bisection starts from the naive predictor and
        walks in.

        Bisected before it is polished because the equation is odd -- convex on
        one side of the origin, concave on the other -- so a Newton step from the
        concave side overshoots, and the rate law amplifies one overshoot into an
        overflow on the following sweep.
        """
        zero = c * 0.0
        lo, hi = zero, zero + 1.0
        # Orient the bracket residual increasing in t, whatever the sign of c.
        s = where(lt(c, zero), zero - 1.0, zero + 1.0)

        for _ in range(self.bisections):
            mid = (lo + hi) * 0.5
            w = mid * c
            f = s * (w + aii * self._phi_inv(w, pt) - c)
            neg = lt(f, zero)
            lo = where(neg, mid, lo)
            hi = where(neg, hi, mid)
        w = (lo + hi) * 0.5 * c

        # Newton polish off the rate law's own tangent:
        # d/dw [w + aii*phi_inv(w)] = 1 + aii * dphi_inv/dw. Safe here because
        # bisection has already put the iterate at the root.
        for _ in range(self.polish):
            _, jvp = self.rate_law.jvp(
                {self.driving_force_input: w, **pt},
                {self.driving_force_input: w * 0.0 + 1.0},
            )
            dg = next(iter(jvp.values()))
            g = self._phi_inv(w, pt)
            w = w - (w + aii * g - c) / (aii * dg + 1.0)
        return self._phi_inv(w, pt)

    def _slice_pt(self, pt: dict[str, Scalar], i: int, m: int) -> dict[str, Scalar]:
        """The rate law's other inputs at component *i*.

        A per-component input (the slip strengths) is indexed; anything carried
        once for the whole vector is passed through untouched.
        """
        return {
            n: (v.sub_batch[i] if v.sub_batch_shape and v.sub_batch_shape[-1] == m else v)
            for n, v in pt.items()
        }

    def _sweep(
        self, g: list[Scalar], bs: list[Scalar], A: Scalar, pt: dict[str, Scalar], m: int
    ) -> list[Scalar]:
        """One Gauss-Seidel sweep over the components.

        Sequential, not simultaneous: only Gauss-Seidel is coordinate descent and
        therefore monotone in the potential. Jacobi is not a descent method and
        was measured to diverge outright on crystal plasticity, where the
        off-diagonal coupling rivals the diagonal.

        The components are held in a plain list, so the sequential dependence
        costs nothing and no tensor is written in place -- the vector is
        reassembled by :func:`~neml2.types.stack` at the sweep boundary. That is
        deliberate: index writes are a friction point for ``torch.export`` and a
        stack is not, and this is the unit the compiled route exports, with the
        sweep loop itself running in the C++ runtime as Newton's does.
        """
        for i in range(m):
            c = bs[i]
            for j in range(m):
                if j != i:
                    c = c - A.sub_batch[i, j] * g[j]
            g[i] = self._coord_solve(A.sub_batch[i, i], c, self._slice_pt(pt, i, m))
        return g

    def forward(  # type: ignore[override]
        self,
        *args: Scalar,
        v: ChainRuleDict | None = None,
    ):
        del v  # a warm start is one-shot and is never differentiated
        vals = dict(zip(self.input_spec, args, strict=False))
        A, b = vals[self._A], vals[self._b]
        pt = {n: vals[n] for n in self._passthrough}
        m = int(b.sub_batch_shape[-1])

        bs = [b.sub_batch[i] for i in range(m)]
        g = [x * 0.0 for x in bs]
        for _ in range(self.sweeps):
            g = self._sweep(g, bs, A, pt, m)
        return stack([x.sub_batch for x in g], dim=0)


__all__ = ["CoordinateDescentPredictor"]
