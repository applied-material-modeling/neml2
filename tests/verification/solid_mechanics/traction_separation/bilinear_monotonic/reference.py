#!/usr/bin/env python

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

"""
Generate the analytical reference solution for BiLinearMixedModeTractionSeparation
under a monotonic mixed-mode opening ramp with the BK propagation criterion.

Closed-form damage:
    delta_n0     = N / K
    delta_s0     = S / K
    beta         = sqrt(delta_s1**2 + delta_s2**2) / delta_n           (opening branch)
    delta_init   = delta_n0 * delta_s0 * sqrt(1 + beta**2)
                   / sqrt(delta_s0**2 + beta**2 * delta_n0**2)
    delta_final  = (2 / (K * delta_init)) * (GIc + (GIIc - GIc) * (beta**2 / (1 + beta**2))**eta)
    delta_m      = sqrt(<delta_n>+**2 + delta_s1**2 + delta_s2**2)
    d            = delta_final * (delta_m - delta_init) / (delta_m * (delta_final - delta_init))
                   clamped to [0, 1]
    T            = (1 - d) * K * [<delta_n>+, delta_s1, delta_s2] + K * [<delta_n>-, 0, 0]

Under a strictly monotonic ramp d is non-decreasing, so the irreversibility check
d = max(d_trial, d_old) is satisfied by d_trial alone — the analytical formula
matches the model's per-step output exactly.
"""

import numpy as np

# ---- Parameters (must match bilinear_monotonic.i) ----
K = 1000.0
GIc = 1.0
GIIc = 2.0
N = 10.0
S = 15.0
eta = 2.0

# Mixed-mode opening ramp: starts below delta_init (elastic), passes through the
# softening branch, and ends past delta_final (fully degraded). The peak jump is
# chosen so that delta_m > delta_final at the last step, exercising the saturated
# branch as well as the linear softening one.
delta_max = np.array([0.20, 0.16, 0.12])
times = np.linspace(0.0, 1.0, 41)


def jump(t):
    return delta_max[None, :] * t[:, None]


def traction(delta):
    delta_n = delta[:, 0]
    delta_s1 = delta[:, 1]
    delta_s2 = delta[:, 2]

    delta_n_pos = np.maximum(delta_n, 0.0)
    delta_n_neg = delta_n - delta_n_pos
    delta_s = np.sqrt(delta_s1 * delta_s1 + delta_s2 * delta_s2)

    delta_n0 = N / K
    delta_s0 = S / K

    # Opening branch: beta = delta_s / delta_n; pure-shear branch when delta_n <= 0.
    open_mask = delta_n > 0.0
    beta = np.where(open_mask, delta_s / np.where(open_mask, delta_n_pos, 1.0), 0.0)
    beta_sq = beta * beta

    delta_mixed_init = np.sqrt(delta_s0 ** 2 + beta_sq * delta_n0 ** 2)
    delta_init_open = delta_n0 * delta_s0 * np.sqrt(1.0 + beta_sq) / delta_mixed_init
    delta_init = np.where(open_mask, delta_init_open, delta_s0)

    beta_sq_ratio = beta_sq / (1.0 + beta_sq)
    delta_final_open = (2.0 / (K * delta_init)) * (
        GIc + (GIIc - GIc) * beta_sq_ratio ** eta
    )
    delta_final_default = np.sqrt(2.0) * 2.0 * GIIc / S
    delta_final = np.where(open_mask, delta_final_open, delta_final_default)

    delta_m = np.sqrt(delta_n_pos ** 2 + delta_s1 ** 2 + delta_s2 ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        d_linear = delta_final * (delta_m - delta_init) / (delta_m * (delta_final - delta_init))
    d = np.where(delta_m <= delta_init, 0.0, np.where(delta_m >= delta_final, 1.0, d_linear))

    factor = (1.0 - d) * K
    T_active_n = factor * delta_n_pos
    T_active_s1 = factor * delta_s1
    T_active_s2 = factor * delta_s2
    T_n = T_active_n + K * delta_n_neg
    return np.column_stack([T_n, T_active_s1, T_active_s2])


delta = jump(times)
T = traction(delta)

# Sanity check: at peak load the trajectory has crossed delta_final and damage saturates to 1,
# so the active traction is (1 - 1) * K * delta = 0 (no compression in this scenario).
np.testing.assert_allclose(T[-1], np.zeros(3), atol=1e-12)

header = ["time", "delta_n", "delta_s1", "delta_s2", "T_n", "T_s1", "T_s2"]
data = np.column_stack([times, delta, T])
np.savetxt(
    "reference.csv",
    data,
    delimiter=",",
    header=",".join(header),
    comments="",
    fmt="%.10e",
)
print(f"wrote reference.csv with {len(times)} rows")
