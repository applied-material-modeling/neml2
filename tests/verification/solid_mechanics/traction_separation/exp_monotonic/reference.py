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
Generate the analytical reference solution for ExpTractionSeparation under a monotonic
mixed-mode opening ramp. The closed-form response is

    delta_eff = sqrt(delta_n**2 + beta * (delta_s1**2 + delta_s2**2) + eps)
    kappa     = max history of delta_eff           (irreversible_damage = true)
    d         = 1 - exp(-kappa / delta_0)
    T_i       = (1 - d) * (Gc / delta_0**2) * delta_i

Under a strictly monotonic ramp the history maximum coincides with the current value, so
kappa = delta_eff and the damage history drops out of the analytical comparison. The result
is written to reference.csv for use by exp_monotonic.i / VTestVerification.
"""

import numpy as np

# ---- Parameters (must match exp_monotonic.i) ----
Gc = 2.0
delta_0 = 1.0
beta = 1.0
eps_reg = 1e-12

# Monotonic mixed-mode opening ramp into the softening tail.
delta_max = np.array([3.0, 1.5, 0.0])
times = np.linspace(0.0, 1.0, 41)


def jump(t):
    return delta_max[None, :] * t[:, None]


def traction(delta):
    delta_n = delta[:, 0]
    delta_s1 = delta[:, 1]
    delta_s2 = delta[:, 2]
    delta_eff = np.sqrt(
        delta_n * delta_n + beta * (delta_s1 * delta_s1 + delta_s2 * delta_s2) + eps_reg
    )
    d = 1.0 - np.exp(-delta_eff / delta_0)
    factor = (1.0 - d) * (Gc / (delta_0 * delta_0))
    return factor[:, None] * delta


delta = jump(times)
T = traction(delta)

# Sanity check: at peak load the damage is 1 - exp(-delta_eff_max/delta_0); the traction
# magnitude should be (1-d) * (Gc/delta_0**2) * |delta|.
delta_eff_max = np.sqrt(delta_max @ delta_max + eps_reg * 0.0)
d_max = 1.0 - np.exp(-delta_eff_max / delta_0)
T_expected_last = (1.0 - d_max) * (Gc / delta_0 ** 2) * delta_max
np.testing.assert_allclose(T[-1], T_expected_last, rtol=1e-10, atol=1e-12)

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
