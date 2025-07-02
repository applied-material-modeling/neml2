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

import torch
from matplotlib import pyplot as plt

# Load the saved model
result = torch.jit.load("result.pt")

I = dict(result.input.named_buffers())
O = dict(result.output.named_buffers())

expected = 14832

nstep = 100

T = [I["{}.forces/T".format(i)].item() for i in range(nstep)][1:]  # skip first step
t = [I["{}.forces/t".format(i)].item() for i in range(nstep)][1:]

q = [O["{}.state/q".format(i)].item() for i in range(nstep)][1:]
eta = [O["{}.state/eta".format(i)].item() for i in range(nstep)][1:]
phif_l = [O["{}.state/phif_l".format(i)].item() for i in range(nstep)][1:]
phif_s = [O["{}.state/phif_s".format(i)].item() for i in range(nstep)][1:]

# takes the integral of qdt using trapezoidal rule
heat_release = -0.5 * sum((t[i] - t[i + 1]) * (q[i + 1] + q[i]) for i in range(nstep - 2))

assert (
    abs(heat_release - expected) < 1
), f"Heat release {heat_release} does not match expected {expected}"

plt.plot(T, q, "ko-")
plt.xlabel("Temperature (K)")
plt.ylabel("Solidification fraction")
plt.grid()
plt.tight_layout()
plt.show()
