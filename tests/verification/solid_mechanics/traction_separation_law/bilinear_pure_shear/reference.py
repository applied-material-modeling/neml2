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
Generate the analytical reference solution for CamanhoDavilaTraction (or any other
BilinearMixedModeTraction subclass) under pure-shear loading with delta_n = 0.

In the pure-shear / compression branch (dn <= 0) the BilinearMixedModeTraction
defaults are used:

    delta_init  = S / K
    delta_final = 2 * GIIc / S
    d           = delta_final * (delta_m - delta_init) / (delta_m * (delta_final - delta_init))
                  clamped to [0, 1]
    T_n         = 0                      (no normal jump, no compression)
    T_s         = (1 - d) * K * delta_s

Under a strictly monotonic ramp d is non-decreasing, so the irreversibility cap
d = max(d_trial, d_old) is satisfied by d_trial alone — this analytical formula
matches the model output step-for-step. The integrated dissipation W = sum_i
0.5 (T_i + T_{i-1}) . (delta_i - delta_{i-1}) along this trajectory equals GIIc
exactly for the closed form; that is checked at the end as a sanity test and
verified externally by verify_dissipation.py against the model output.
"""

import numpy as np

# ---- Parameters (must match bilinear_pure_shear.i) ----
K = 1000.0
GIIc = 2.0
S = 15.0

delta_init = S / K
delta_final = 2.0 * GIIc / S

# Pure-shear ramp on delta_s1; delta_n = delta_s2 = 0 throughout. Ramp goes from 0 to
# 1.5 * delta_final so the trajectory exercises the elastic branch, the bilinear
# softening branch, and the saturated-damage branch.
nstep = 200
times = np.linspace(0.0, 1.0, nstep)
delta_s1_max = 0.4

delta_n = np.zeros(nstep)
delta_s1 = delta_s1_max * times
delta_s2 = np.zeros(nstep)

delta_m = np.abs(delta_s1)
with np.errstate(divide="ignore", invalid="ignore"):
    d_linear = delta_final * (delta_m - delta_init) / (delta_m * (delta_final - delta_init))
d = np.where(delta_m <= delta_init, 0.0, np.where(delta_m >= delta_final, 1.0, d_linear))

T_n = np.zeros(nstep)
T_s1 = (1.0 - d) * K * delta_s1
T_s2 = np.zeros(nstep)

# Sanity check: integrated work along the trajectory equals GIIc (energy conservation
# of the bilinear pure-shear law).
dW = 0.5 * (T_s1[1:] + T_s1[:-1]) * (delta_s1[1:] - delta_s1[:-1])
W = float(dW.sum())
np.testing.assert_allclose(W, GIIc, rtol=1e-2)

header = ["time", "delta_n", "delta_s1", "delta_s2", "T_n", "T_s1", "T_s2"]
data = np.column_stack([times, delta_n, delta_s1, delta_s2, T_n, T_s1, T_s2])
np.savetxt(
    "reference.csv",
    data,
    delimiter=",",
    header=",".join(header),
    comments="",
    fmt="%.10e",
)
print(f"wrote reference.csv with {len(times)} rows")
