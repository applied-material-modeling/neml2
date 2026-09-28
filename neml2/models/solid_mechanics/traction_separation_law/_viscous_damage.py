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

r"""Damage irreversibility cap and viscous regularization for the TSLs.

Both damage-bearing cohesive laws here evolve the same pair of internal
variables -- a trial damage $d_t$ from the law's own envelope formula, and its
previous-step value $d_{n-1}$ (the ``~1`` history of the ``damage`` output) --
and then subject that pair to the same two steps, in order: an irreversibility
cap $d^* = \max(d_t, d_{n-1})$, then the backward-Euler viscosity regularization
that MOOSE's ``BilinearMixedModeCohesiveZoneModel`` applies to its damage.

Only the envelope differs between the bilinear and the Salehani-Irani laws, so
it stays with each law while this step lives here once. A plain function rather
than a registered :class:`~neml2.models.model.Model`: the step has no
user-facing HIT surface and is not meant to be swapped from an input file, so a
schema would only add a place for the metadata to be wrong. Were a user to want
a different regularization -- Perzyna overstress instead of a linear dashpot --
that is a genuine substitution and belongs in a ``Model`` leaf the dependency
resolver can wire between the damage formula and the traction assembly.

The regularization is skipped for zero ``eta``, which keeps the rate-independent
update exact and avoids the $\eta = \Delta t = 0$ indeterminate form of a
transient's first step. A negative ``eta`` is rejected outright by
:func:`check_viscosity` at model-construction time rather than quietly reading
as inviscid.
"""

from __future__ import annotations

from dataclasses import dataclass

from ....types import Scalar, gt, where


@dataclass(frozen=True)
class ViscousDamage:
    """Regularized damage plus the pushforward coefficients it was built from.

    The coefficients are the partials of :attr:`d` with respect to the
    inputs that :func:`viscous_damage` consumed, so a caller assembling a
    traction from :attr:`d` can thread them straight into its own chain-rule
    actions without re-deriving them.
    """

    #: Regularized damage, the value to feed the traction assembly.
    d: Scalar
    #: Mask of the points where the cap advanced, i.e. $d_t > d_{n-1}$.
    advance: Scalar
    #: Blending factor $\Delta t / (\eta + \Delta t)$, or one on the
    #: inviscid branch. Multiplies every $\partial d / \partial d_t$-style
    #: partial the caller derives from its own envelope.
    alpha: Scalar
    #: $\partial d / \partial d_{n-1}$, routing the cap's own dependence on
    #: the previous-step value.
    dd_dd_old: Scalar
    #: $\partial d / \partial t$, so $\partial d / \partial t_{n-1}$ is its
    #: negation.
    dd_dt: Scalar


def viscous_damage(
    d_trial: Scalar,
    d_old: Scalar,
    t: Scalar,
    t_old: Scalar,
    eta: Scalar,
) -> ViscousDamage:
    r"""Cap *d_trial* for irreversibility, then regularize it with viscosity.

    Computes

    .. math::
        d^* = \max(d_t, d_{n-1}), \qquad
        d_n = d_{n-1} + \frac{\Delta t}{\eta + \Delta t}(d^* - d_{n-1}),

    with $\Delta t = t - t_{n-1}$. A zero ``eta`` takes the rate-independent
    branch: :attr:`ViscousDamage.d` is then exactly the capped value and
    :attr:`ViscousDamage.alpha` is one. The branch is a ``where`` on ``eta > 0``
    rather than a Python ``if`` because ``eta`` is a tensor that may carry
    per-point values; :func:`check_viscosity` is what rejects the one sign that
    has no reading at all.

    Parameters
    ----------
    d_trial
        Inviscid damage from the law's own envelope formula.
    d_old
        Previous-step damage.
    t
        Time at the current step.
    t_old
        Time at the previous step.
    eta
        Damage viscosity. Non-negative; zero for the rate-independent law.
    """
    one = Scalar.from_value(1.0, like=d_trial)
    zero = Scalar.from_value(0.0, like=d_trial)

    # Irreversibility cap. The mask is a bool tensor (no grad), matching the
    # C++ ``advance_mask.detach()`` -- damage never un-advances.
    advance = gt(d_trial, d_old)
    d_inviscid = where(advance, d_trial, d_old)

    # Backward-Euler regularization. The branch on ``eta > 0`` is what keeps
    # the first transient step safe: there ``eta = dt = 0`` and the direct
    # ``dt / (eta + dt)`` would be 0/0.
    dt = t - t_old
    eta_positive = gt(eta, 0.0)
    denom = where(eta_positive, eta + dt, one)
    alpha = where(eta_positive, dt / denom, one)
    d = d_old + alpha * (d_inviscid - d_old)

    # d = (1 - alpha) d_old + alpha d^*, and d^*'s own dependence on d_old is
    # the (detached) mask: zero where the cap advances, one where it freezes.
    dd_dd_old = (one - alpha) + alpha * where(advance, zero, one)
    # d(alpha)/d(t) = eta / (eta + dt)^2, since t = dt + t_old.
    dd_dt = where(eta_positive, eta / (denom * denom) * (d_inviscid - d_old), zero)

    return ViscousDamage(
        d=d,
        advance=advance,
        alpha=alpha,
        dd_dd_old=dd_dd_old,
        dd_dt=dd_dt,
    )


def check_viscosity(eta: Scalar, *, owner: str) -> None:
    r"""Reject a negative damage viscosity at model-construction time.

    A negative ``eta`` is not a weaker regularization, it is a sign error: the
    dashpot would run backwards, and $\Delta t / (\eta + \Delta t)$ is not even
    bounded on the inviscid side of it. Left to the tensor branch in
    :func:`viscous_damage` it reads as *rate independent*, so a mistyped ``-1``
    would silently turn a viscous law into a rate-dependent one and produce
    plausible-looking damage. Failing here puts the error where the coefficient
    was written.

    Parameters
    ----------
    eta
        The law's ``viscosity`` parameter. Read once, on the host, at
        construction -- see :meth:`~neml2.types.TensorWrapper.item` for why
        that is not something a ``forward`` may do. A viscosity carrying batch
        or sub-batch axes is left unchecked: the per-point reading is only
        decidable on device, and the ``eta > 0`` branch in
        :func:`viscous_damage` already treats each point on its own.
    owner
        Name of the model being validated, for the message.
    """
    if eta.ndim != 0:
        return
    value = eta.item()
    if value < 0.0:
        raise ValueError(
            f"{owner}: viscosity must be non-negative, got {value}. A negative "
            "viscosity has no physical reading -- it reverses the dashpot, and "
            "dt / (viscosity + dt) is unbounded once dt < -viscosity. Use 0 for "
            "the rate-independent law."
        )
