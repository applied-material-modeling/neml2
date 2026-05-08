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
Generate the analytical reference solution for PureElasticTractionSeparation under a mixed-mode
linear ramp of the displacement jump. The traction is the closed-form

    T_n  = K_n * delta_n
    T_s1 = K_t * delta_s1
    T_s2 = K_t * delta_s2

evaluated at every prescribed time point and written to reference.csv.
"""

import numpy as np

# ---- Parameters (must match pure_elastic.i) ----
K_n = 1000.0
K_t = 500.0

# Maximum displacement-jump components (Vec order: normal, shear-1, shear-2)
delta_max = np.array([0.05, 0.02, -0.01])

# Linear ramp on a uniform grid: pure-elastic response has no internal time scale, so the time
# grid is purely a parametrization of the load history.
times = np.linspace(0.0, 1.0, 31)


def jump(t):
    return delta_max[None, :] * t[:, None]


def traction(delta):
    K = np.array([K_n, K_t, K_t])
    return K[None, :] * delta


delta = jump(times)
T = traction(delta)

# Sanity check: linearity. T(t) = K * delta_max * t.
np.testing.assert_allclose(T[-1], np.array([K_n, K_t, K_t]) * delta_max, rtol=1e-12, atol=1e-12)

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
