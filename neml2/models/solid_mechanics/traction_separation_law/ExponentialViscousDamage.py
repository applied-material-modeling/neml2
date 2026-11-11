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

"""Exact exponential viscous damage update."""

from ....factory import register_neml2_object
from ....types import Scalar, exp, where
from .ViscousDamageUpdate import ViscousDamageUpdate


@register_neml2_object("ExponentialViscousDamage")
class ExponentialViscousDamage(ViscousDamageUpdate):
    r"""Irreversible update with $\alpha=1-\exp(-\Delta t/\eta)$."""

    def _coefficient(
        self, dt: Scalar, eta: Scalar, positive: Scalar, one: Scalar, zero: Scalar
    ) -> tuple[Scalar, Scalar]:
        safe_eta = where(positive, eta, one)
        decay = exp(-dt / safe_eta)
        alpha = where(positive, one - decay, one)
        derivative = where(positive, decay / safe_eta, zero)
        return alpha, derivative


__all__ = ["ExponentialViscousDamage"]
