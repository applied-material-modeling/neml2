#!/usr/bin/env python
"""
Generate the analytical reference solution for the Zener (Standard Linear Solid) viscoelastic
model under a linear strain ramp followed by a hold, written to reference.csv for use by
zener.i / VTestVerification.

The Zener model is an equilibrium spring (E_inf) in parallel with a Maxwell branch (E_M, eta_M).
Letting tau = eta_M / E_M and treating each tensor component independently:

    sigma(t)   = (E_inf + E_M) * eps(t) - E_M * eps_v(t)
    deps_v/dt  = (1/tau) * (eps - eps_v)

For a linear ramp eps(t) = eps0 * t / T_ramp on [0, T_ramp] and hold eps(t) = eps0 for t > T_ramp,
with eps_v(0) = 0, the closed-form viscous strain is:

  During ramp, 0 <= t <= T_ramp:
    eps_v(t) = (eps0 / T_ramp) * t - (eps0 * tau / T_ramp) * (1 - exp(-t/tau))

  After ramp, t > T_ramp:
    eps_v(t) = eps0 - (eps0 * tau / T_ramp) * (exp(T_ramp/tau) - 1) * exp(-t/tau)

Stress follows from the algebraic relation above. CSV columns are written in NEML2's Mandel
SR2 order: xx, yy, zz, yz, xz, xy.
"""

import numpy as np

# ---- Parameters (must match zener.i) ----
E_inf = 1000.0
E_M = 5000.0
eta_M = 100.0
tau = eta_M / E_M  # 0.02

# Strain target (Mandel components: xx, yy, zz, yz, xz, xy).  Uniaxial-like, deviatoric only.
eps0 = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])

T_ramp = 0.05  # ~ 2.5 * tau
t_final = 1.0  # ~ 50 * tau, well into the long-time regime

# Time grid: linear during the ramp (uniform driving), then geometrically spaced afterwards so
# dt grows smoothly through the relaxation. A naive linear-then-coarse grid causes a step where
# dt/tau jumps from O(0.1) to O(few), which a first-order backward-Euler integrator cannot
# resolve even though the underlying physics is essentially at steady state.
times = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, T_ramp, 51),  # ramp: dt/tau = 0.05
            T_ramp + np.geomspace(tau / 20, t_final - T_ramp, 60),  # smooth log tail
        ]
    )
)


def strain(t):
    """Imposed strain history."""
    return np.where((t <= T_ramp)[:, None], eps0[None, :] * (t[:, None] / T_ramp), eps0[None, :])


def viscous_strain(t):
    """Analytical viscous strain ε_v(t) for each tensor component."""
    ramp_phase = (eps0[None, :] / T_ramp) * t[:, None] - (eps0[None, :] * tau / T_ramp) * (
        1.0 - np.exp(-t[:, None] / tau)
    )
    hold_phase = eps0[None, :] - (eps0[None, :] * tau / T_ramp) * (
        np.exp(T_ramp / tau) - 1.0
    ) * np.exp(-t[:, None] / tau)
    return np.where((t <= T_ramp)[:, None], ramp_phase, hold_phase)


def stress(eps, eps_v):
    """sigma = (E_inf + E_M) * eps - E_M * eps_v, element-wise."""
    return (E_inf + E_M) * eps - E_M * eps_v


eps = strain(times)
eps_v = viscous_strain(times)
sig = stress(eps, eps_v)

# Sanity check: at t = t_final (well past 5*tau), sigma should approach E_inf * eps0.
np.testing.assert_allclose(sig[-1], E_inf * eps0, rtol=1e-3, atol=1e-6)

# CSV columns: time, then 6 strain components, then 6 stress components, all Mandel order.
header = ["time", "exx", "eyy", "ezz", "eyz", "exz", "exy", "sxx", "syy", "szz", "syz", "sxz", "sxy"]
data = np.column_stack([times, eps, sig])
np.savetxt(
    "reference.csv",
    data,
    delimiter=",",
    header=",".join(header),
    comments="",
    fmt="%.10e",
)
print(f"wrote reference.csv with {len(times)} rows")
