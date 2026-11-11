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

"""Wiring check for the ``lumped_km_hardening`` regression scenario.

One ``KocksMeckingRateSensitivity`` is the single source of the rate
sensitivity n: it feeds ``SmoothCap`` -> ``PowerLawSlipRule`` (capped) and
``KocksMeckingDynamicRecovery`` (uncapped). The composed model must therefore
expose exactly one Kocks-Mecking ``A`` parameter, and neither the slip-rule
exponent nor the forest-evolution ``k2`` may survive as a constant
parameter -- both are driven by upstream models.
"""

from __future__ import annotations

from pathlib import Path

import neml2

_INPUT = (
    Path(__file__).parents[1]
    / "regression"
    / "solid_mechanics"
    / "crystal_plasticity"
    / "lumped_km_hardening"
    / "model.i"
)


def test_single_kocks_mecking_A():
    model = neml2.load_model(str(_INPUT), "model")
    names = [name for name, _ in model.named_parameters()]
    assert [n for n in names if n.split(".")[-1] == "A"] == [
        "update.implicit_rate.rate_sensitivity.A"
    ]
    assert not any(n.endswith("slip_rule.n") for n in names)
    assert not any(n.endswith("dislocation_density_rate.k2") for n in names)
