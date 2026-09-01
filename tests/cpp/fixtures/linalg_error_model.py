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

"""External (out-of-package) NEML2 model whose forward raises
``torch.linalg.LinAlgError`` from a real linalg kernel (Cholesky of a zero
matrix). Used by ``test_eager.cpp`` to drive the eager runtime's
``is_torch_linalg_error`` -> recoverable ``ConvergenceError`` promotion path
across the embedded-Python boundary.
"""

from __future__ import annotations

import torch

from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import SR2


@register_neml2_object("SingularLinAlgFailure")
class SingularLinAlgFailure(Model):
    """A model whose forward always trips a ``torch.linalg.LinAlgError``.

    ``torch.linalg.cholesky`` on an all-zero matrix is not positive-definite and
    raises ``torch.linalg.LinAlgError`` directly from the linalg kernel -- the
    Python face of the ``c10::LinAlgError`` the aoti guard also promotes to
    ``ConvergenceError``. The eager guard sees this as a ``py::error_already_set``
    matching ``torch.linalg.LinAlgError`` and re-raises it as the recoverable
    ``neml2::aoti::ConvergenceError``.
    """

    hit = HitSchema(
        input("in_stress", SR2, "Input stress (ignored)"),
        output("out_stress", SR2, "Output stress (unreachable)"),
    )

    def forward(  # type: ignore[override]
        self,
        in_stress: SR2,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        # Build a zero PSD matrix on the same device/dtype as the input, then
        # ask torch for its Cholesky factor. Zero has no positive-definite
        # factorization, so this raises torch.linalg.LinAlgError.
        A = torch.zeros((2, 2), dtype=in_stress.data.dtype, device=in_stress.data.device)
        torch.linalg.cholesky(A)
        raise RuntimeError("unreachable: cholesky was expected to raise")  # pragma: no cover
