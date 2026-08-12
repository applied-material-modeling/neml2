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

$A$ arrives two ways. Where a closed form exists it is cheapest --
``SlipSystemElasticInteraction`` gives crystal plasticity's
$\Delta t\,M^{\mathsf T}\mathbb{C}M$ in one block, and deliberately drops the spin
convection so that $A$ stays symmetric. Otherwise
:class:`~neml2.models.common.RateCondensation` differentiates the authored return
path at zero rate, which is exact for any hardening law and costs no per-physics
code. Viscoplasticity takes the second route: writing $A$ out there means one term
per hardening mechanism, and freezing them instead is not accurate enough (see
that class).

Measured on a single-crystal scenario, warm-starting the *rate* (non-inverted)
formulation this way takes the first time step from 16 Newton iterations to 5,
and widens the convergence basin fourfold. Five is one above the floor: seeding
the *exact converged* values of the predicted unknowns still costs 4, because
that seed leaves the remaining unknowns cold and so is not at the root either.
Later steps are untouched -- the prediction is gated to the cold start, and
applying it warm costs iterations rather than saving them.

Viscoplasticity is the one-coordinate case, and there the condensation is exact
for a linear hardening law: with $A$ from
:class:`~neml2.models.common.RateCondensation`, the first step converges *at the
predictor* -- zero Newton iterations -- across six decades of step size, against
15 with no predictor and 6-9 for the inverted (flow-rate-as-unknown) formulation.
Chaboche, whose return path is genuinely nonlinear over a step, lands at 2-3.

This model outputs the **rate**. Converting that into whatever unknowns the
implicit system actually carries -- for crystal plasticity, the elastic strain
and the slip hardening -- is left to ordinary model composition downstream, so
no physics enters this class.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ...factory import register_neml2_object
from ...schema import HitSchema, dependency, input, option, output
from ...types import Scalar, lt, stack, where
from ..model import IterableExport, Model

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

    A system with a single coordinate is written without the sub-batch axes, as
    plain scalars. Viscoplasticity is that case: the update condenses onto one
    flow rate, so $m = 1$ and the sweep is a single exact scalar solve. The two
    layouts are told apart by sub-batch rank and a mismatched pair is rejected.
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
            "semi-definite -- or a plain scalar for a one-coordinate system. Its "
            "diagonal must be non-negative -- that is what brackets each coordinate "
            "solve.",
            attr="_A",
        ),
        input(
            "trial_driving_force",
            Scalar,
            "The right-hand side $b$, sub-batched over (m,) -- or a plain scalar for a "
            "one-coordinate system: the driving force that would act with every rate "
            "at zero (for crystal plasticity, the trial resolved shear; for "
            "viscoplasticity, the trial yield function).",
            attr="_b",
        ),
        output(
            "rate",
            Scalar,
            "The predicted rate, laid out like `trial_driving_force`",
            attr="_g",
        ),
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
    #: Set only on the single-sweep copy built by `iterable_export_form`.
    _feedback_in: str

    def __init__(
        self,
        *,
        rate_law: Model,
        driving_force_input: str,
        sweeps: int = 16,
        bisections: int = 16,
        polish: int = 3,
        **hit_values,
    ) -> None:
        # ``**hit_values`` is load-bearing, not boilerplate. This model's
        # ``input_spec`` is only knowable at construction -- it depends on which
        # rate law is plugged in -- so it has to be extended below. That edit
        # survives only if schema resolution has already happened, and
        # ``Model.from_hit`` decides *when* it happens from this very signature:
        # with ``**kwargs`` it constructs once and ``super().__init__()``
        # resolves the schema first; without it, the remaining schema fields are
        # applied by a ``_store_schema_values`` call *after* ``__init__``, which
        # rebuilds ``input_spec`` from the class-level spec and drops whatever
        # was appended here. Named-only signatures are therefore fine for a
        # static leaf and quietly wrong for a dynamic one.
        super().__init__(**hit_values)
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
        self.input_spec = {
            **self.input_spec,
            **{n: rate_law.input_spec[n] for n in self._passthrough},
        }

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

    # -- system layout ------------------------------------------------
    #
    # Two layouts are accepted, distinguished by whether the driving force
    # carries a sub-batch axis:
    #
    # * ``b`` over ``(m,)`` and ``A`` over ``(m, m)`` -- the vector system.
    #   Crystal plasticity: one coordinate per slip system.
    # * ``b`` and ``A`` plain scalars -- the degenerate one-coordinate system.
    #   Viscoplasticity: the whole update condenses onto a single flow rate, so
    #   there is nothing to sub-batch over and Gauss-Seidel is one exact scalar
    #   solve. Demanding a length-1 sub-batch axis here would mean inventing a
    #   model whose only job is to unsqueeze one.

    @staticmethod
    def _dimension(b: Scalar) -> int:
        return 1 if b.sub_batch_ndim == 0 else int(b.sub_batch_shape[-1])

    @staticmethod
    def _components(b: Scalar) -> list[Scalar]:
        if b.sub_batch_ndim == 0:
            return [b]
        return [b.sub_batch[i] for i in range(int(b.sub_batch_shape[-1]))]

    @staticmethod
    def _entry(A: Scalar, i: int, j: int) -> Scalar:
        return A if A.sub_batch_ndim == 0 else A.sub_batch[i, j]

    @staticmethod
    def _assemble(g: list[Scalar], b: Scalar) -> Scalar:
        if b.sub_batch_ndim == 0:
            return g[0]
        return stack([x.sub_batch for x in g], dim=0)

    @staticmethod
    def _check_layout(A: Scalar, b: Scalar) -> None:
        """Reject a mismatched ``(A, b)`` pair before it reads as garbage.

        Sub-batch rank is the only thing distinguishing the two layouts, so a
        mis-wired input -- an ``A`` that forgot its second axis, say -- would
        otherwise index into the wrong axis and silently produce a plausible
        number.
        """
        want = 0 if b.sub_batch_ndim == 0 else 2
        if A.sub_batch_ndim != want or b.sub_batch_ndim > 1:
            raise ValueError(
                f"CoordinateDescentPredictor: the driving force {b.sub_batch_ndim} and coupling "
                f"{A.sub_batch_ndim} sub-batch ranks do not form a system. Expected either "
                f"(1, 2) for a vector system over (m,) and (m, m), or (0, 0) for a single "
                f"coordinate."
            )

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
                    c = c - self._entry(A, i, j) * g[j]
            g[i] = self._coord_solve(self._entry(A, i, i), c, self._slice_pt(pt, i, m))
        return g

    def _sweeps_from(self, g: list[Scalar], vals: dict[str, Scalar], n: int) -> Scalar:
        """Run *n* Gauss-Seidel sweeps starting from *g*, and reassemble."""
        A, b = vals[self._A], vals[self._b]
        self._check_layout(A, b)
        pt = {name: vals[name] for name in self._passthrough}
        m = self._dimension(b)
        bs = self._components(b)
        for _ in range(n):
            g = self._sweep(g, bs, A, pt, m)
        return self._assemble(g, b)

    def _unpack(self, args) -> dict[str, Scalar]:
        # strict: a short pack means the caller resolved fewer inputs than this
        # model declares, which is a wiring bug. Silently truncating turns it
        # into a KeyError several lines later, pointing at the wrong thing.
        return dict(zip(self.input_spec, args, strict=True))

    def forward(  # type: ignore[override]
        self,
        *args: Scalar,
        v: ChainRuleDict | None = None,
    ):
        del v  # a warm start is one-shot and is never differentiated
        vals = self._unpack(args)
        b = vals[self._b]
        zero = [c * 0.0 for c in self._components(b)]
        return self._sweeps_from(zero, vals, self.sweeps)

    # ------------------------------------------------------------------
    # Iterable-export protocol
    # ------------------------------------------------------------------

    def iterable_export_form(self) -> IterableExport:
        """One sweep, plus the feedback pair a runtime needs to iterate it.

        The sweep count is a Python ``for`` here, so exporting this model as-is
        unrolls it into the graph: at the defaults that is 16 sweeps x 12
        components x ~19 inner iterations, several thousand copies of the rate
        law. Worse, an unrolled count is frozen at compile time and can never
        stop early.

        So the compiled routes export a **single sweep** and run the loop
        themselves, exactly as they already do for Newton. The single-sweep form
        takes the incoming rate as an extra input and returns the outgoing one,
        and the runtime feeds that output back to that input ``iterations``
        times.

        Note this iterates the *enclosing predictor graph*, not this leaf alone,
        so the trial state and the back-substitution around it recompute on
        every sweep. That is deliberate: splitting the predictor into pre/sweep/
        post graphs would buy back a fraction of an already-cheap stage in
        exchange for a three-way split of the export planner.
        """
        sweep = copy.copy(self)
        # Shallow copy shares submodules (the rate law) but must not share the
        # input_spec dict, which the feedback input extends.
        sweep.sweeps = 1
        feedback_in = f"{self._g}_in"
        sweep.input_spec = {**self.input_spec, feedback_in: Scalar}
        sweep._feedback_in = feedback_in
        sweep.forward = sweep._forward_one_sweep  # type: ignore[method-assign]
        return IterableExport(
            model=sweep,
            feedback_input=feedback_in,
            feedback_output=self._g,
            iterations=self.sweeps,
        )

    def _forward_one_sweep(self, *args: Scalar, v: ChainRuleDict | None = None):
        """One sweep from the incoming rate. See :meth:`iterable_export_form`."""
        del v
        vals = self._unpack(args)
        b = vals[self._b]
        # The feedback arrives as a *graph input*, and `sub_batch_ndim` does not
        # survive the export pytree round-trip -- it is rebuilt at 0, leaving the
        # slip axis looking like batch. `b` is produced inside the graph, so its
        # metadata is intact and says how many trailing axes to re-read.
        g_in = vals[self._feedback_in].with_sub_batch_ndim(b.sub_batch_ndim)
        return self._sweeps_from(self._components(g_in), vals, 1)


__all__ = ["CoordinateDescentPredictor"]
