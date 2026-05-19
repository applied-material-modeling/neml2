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
Generate the analytical reference solution for SalehaniIrani3DCTractionSeparation under a
mixed-mode linear ramp of the displacement jump. The traction is the closed-form

    delta_u0_t_int = sqrt(2) * delta_u0_t                 (internal tangential length)
    a_n = e * Tmax_n
    a_t = sqrt(2 * e) * Tmax_t
    b_n  = delta_n  / delta_u0_n
    b_si = delta_si / delta_u0_t_int
    x    = b_n + b_s1**2 + b_s2**2
    T_n  = a_n * b_n  * exp(-x)
    T_si = a_t * b_si * exp(-x)

evaluated at every prescribed time point and written to reference.csv. The model has no
internal state, so the response is path-independent and the reference is exact.
"""

import numpy as np

# ---- Parameters (must match salehani_irani.i) ----
delta_u0_n = 1.0
delta_u0_t = 1.0  # raw user value; internal value is sqrt(2) * this
Tmax_n = 1.0
Tmax_t = 1.0

delta_u0_t_int = np.sqrt(2.0) * delta_u0_t
a_n = np.e * Tmax_n
a_t = np.sqrt(2.0 * np.e) * Tmax_t

# Mixed-mode opening ramp into the softening tail (well past the peak in each direction).
delta_max = np.array([3.0, 1.5, 0.0])
times = np.linspace(0.0, 1.0, 41)


def jump(t):
    return delta_max[None, :] * t[:, None]


def traction(delta):
    b_n = delta[:, 0] / delta_u0_n
    b_s1 = delta[:, 1] / delta_u0_t_int
    b_s2 = delta[:, 2] / delta_u0_t_int
    x = b_n + b_s1 * b_s1 + b_s2 * b_s2
    e = np.exp(-x)
    T = np.column_stack([a_n * b_n * e, a_t * b_s1 * e, a_t * b_s2 * e])
    return T


delta = jump(times)
T = traction(delta)

# Sanity check: T(0) = 0; in the pure-normal limit (b_s = 0), T_n peaks at b_n = 1 with
# value a_n / e = Tmax_n.
np.testing.assert_allclose(T[0], np.zeros(3), atol=1e-12)
b_n_grid = np.linspace(0.0, 5.0, 1001)
T_n_pure = a_n * b_n_grid * np.exp(-b_n_grid)
i_peak = int(np.argmax(T_n_pure))
np.testing.assert_allclose(b_n_grid[i_peak], 1.0, atol=5e-3)
np.testing.assert_allclose(T_n_pure[i_peak], Tmax_n, rtol=5e-3)

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
