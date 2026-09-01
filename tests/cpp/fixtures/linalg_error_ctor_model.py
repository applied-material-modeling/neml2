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

"""External (out-of-package) NEML2 model whose ``__init__`` raises
``torch.linalg.LinAlgError`` from a real linalg kernel (Cholesky of a zero
matrix), so the failure occurs at model *construction* rather than at
evaluation. Used by ``test_eager.cpp`` to verify that the eager runtime's
constructor path classifies the LinAlgError as ``FatalError`` (config error)
rather than promoting it to the recoverable ``ConvergenceError`` -- cutting
the time step does not fix a bad initial parameter, so construction failures
should not trigger a retry loop.
"""

from __future__ import annotations

import torch

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import SR2


@register_neml2_object("CtorLinAlgFailure")
class CtorLinAlgFailure(Model):
    """A model that fails during ``__init__`` with ``torch.linalg.LinAlgError``.

    The Cholesky factorization of an all-zero matrix fails (the matrix is
    not positive-definite), so ``torch.linalg.cholesky`` raises
    ``torch.linalg.LinAlgError`` directly from the linalg kernel -- the same
    exception the evaluation-time fixture triggers, but during construction.
    """

    hit = HitSchema(
        input("in_stress", SR2, "Input stress (never reached)"),
        output("out_stress", SR2, "Output stress (never reached)"),
    )

    def __init__(self, **hit_values):
        super().__init__(**hit_values)
        # The zero matrix is not positive-definite, so the Cholesky factorization
        # fails and torch.linalg.LinAlgError is raised.
        torch.linalg.cholesky(torch.zeros((2, 2)))

    def forward(  # type: ignore[override]
        self,
        in_stress: SR2,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        raise RuntimeError("unreachable: __init__ was expected to raise")  # pragma: no cover
