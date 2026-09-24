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

"""Mass and volume conservation check for the non-uniform density PBM case.

The finite-volume flux form conserves total mass by construction; the daughter
matrix (see ``gen_p.py``) is additionally designed so total particle volume is
conserved. Both are preserved exactly by backward Euler regardless of step size,
so this asserts them to a tight tolerance across the whole trajectory.
"""

from __future__ import annotations

from pathlib import Path

import torch

from neml2 import load_input

_INPUT = Path(__file__).parent / "conservation_volume.i"


def test_mass_and_volume_conserved():
    dv = torch.ones(10, dtype=torch.float64)
    rho = torch.cat([torch.tensor([0.5, 3.0]), torch.full((8,), 1.5)]).to(torch.float64)

    factory = load_input(_INPUT)
    driver = factory.get_driver("driver")
    driver.run()
    result = driver.result()

    steps = sorted(
        int(k.split(".")[1]) for k in result if k.startswith("output.") and k.endswith(".u")
    )
    assert len(steps) > 1

    masses, volumes = [], []
    for s in steps:
        u = result[f"output.{s}.u"].detach().reshape(-1).to(torch.float64)
        masses.append(float((u * dv).sum()))
        volumes.append(float((u / rho * dv).sum()))

    mass0, vol0 = masses[0], volumes[0]
    assert vol0 > 0.0
    max_mass_dev = max(abs(m - mass0) for m in masses)
    max_vol_dev = max(abs(v - vol0) for v in volumes)
    assert max_mass_dev < 1e-4, f"mass drift {max_mass_dev}"
    assert max_vol_dev < 1e-4, f"volume drift {max_vol_dev}"
