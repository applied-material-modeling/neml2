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

"""Tripwire for upstream PyTorch bug pytorch/pytorch#187907.

AOTInductor cannot lower a reverse-mode-autograd graph through an element-wise op
whose backward SAVES ITS OUTPUT (``sqrt`` / ``exp`` / ``tanh`` / ``reciprocal``),
under ``strict=True`` + a dynamic batch dim + ``trace_autograd_ops``. AOTAutograd
lifts the saved-output activation as a constant with a symbolic batch shape, which
Inductor can neither inline nor serialise. NEML2 works around this by routing the
affected ops through input-recompute ``torch.autograd.Function``s in
``neml2/types/functions.py`` (``_recompute_unary`` + the ``*_ad`` dispatchers) and
the division dunders.

This test exercises the RAW upstream failure (plain ``torch.sqrt`` / ``torch.exp``,
no NEML2) and is marked ``xfail(strict=True)``. While the bug is present each case
raises and the test xfails. The moment a PyTorch upgrade fixes the bug the compile
succeeds, the strict xfail turns into a hard failure, and that is the signal to
DELETE the NEML2 workarounds:

  * ``_recompute_unary`` / ``_ad_safe`` / ``_*_ad`` / ``*_ad`` and the
    ``requires_grad`` branches in ``sqrt`` / ``exp`` / ``tanh`` / ``reciprocal`` /
    ``norm`` / ``unit`` / ``compose`` in ``neml2/types/functions.py``;
  * the division-dunder routing in ``scalar.py`` / ``_primitive.py`` / ``tensor.py``;
  * the clear-error guard ``_compile_param_derivative_graph`` in
    ``neml2/cli/aoti_export.py``.

Each case triggers an Inductor compile, so this lives in the (slow) AOTI suite.
"""

from __future__ import annotations

import pytest
import torch
import torch._dynamo  # for the trace_autograd_ops availability gate below
from torch.export import Dim, export

from neml2._warnings import TORCH_JIT_PY314, ignore_warnings


@pytest.fixture
def _trace_autograd_ops():
    """Enable ``trace_autograd_ops`` (the only config in which ``autograd.grad``
    lowers) for the duration of a test, restoring it afterwards."""
    prev = torch._dynamo.config.trace_autograd_ops
    torch._dynamo.config.trace_autograd_ops = True
    try:
        yield
    finally:
        torch._dynamo.config.trace_autograd_ops = prev


def _grad_module(op):
    class _M(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            leaf = x.clone().requires_grad_(True)
            out = op(leaf)
            (g,) = torch.autograd.grad(out, leaf, grad_outputs=torch.ones_like(out))
            return g.detach()

    return _M()


@pytest.mark.skipif(
    not hasattr(torch._dynamo.config, "trace_autograd_ops"),
    reason="trace_autograd_ops (and reverse-mode-AD lowering) requires torch >= 2.11",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "pytorch/pytorch#187907: AOTInductor cannot lower a reverse-mode-autograd "
        "graph through a saved-output op under strict + dynamic-batch export. When "
        "this XPASSes, upstream is fixed -- delete the recompute-Function workarounds "
        "in neml2/types/functions.py + the division-dunder routing + the clear-error "
        "guard in neml2/cli/aoti_export.py."
    ),
)
@pytest.mark.parametrize("op_name", ["sqrt", "exp"])
def test_saved_output_op_lowers_through_aoti(op_name, _trace_autograd_ops):
    """Plain ``d op(x)/dx`` (reverse-mode) should lower through AOTInductor. It
    does not today (#187907); ``xfail(strict=True)`` flips to a failure on fix."""
    op = getattr(torch, op_name)
    x = torch.rand(2, dtype=torch.float64) + 0.5  # positive, away from 0
    dyn = ({0: Dim("b", min=1, max=1024)},)
    # Both strict export and the Inductor compile run torch internals that, on
    # Python 3.14, reach for torch's own deprecated torch.jit.script(_method) (the
    # mkldnn import + forward-mode JVP-decomp registration) -- neml2 calls neither.
    # Suppress that torch-internal noise across the whole region; the expected
    # #187907 failure from the compile still propagates to xfail.
    with ignore_warnings(TORCH_JIT_PY314):
        ep = export(_grad_module(op), (x,), dynamic_shapes=dyn, strict=True)

        # No symbolic-shaped constant should appear in a correctly-lowering graph.
        consts = {k: tuple(c.shape) for k, c in ep.constants.items() if hasattr(c, "shape")}  # type: ignore[union-attr]
        assert not any(not isinstance(d, int) for s in consts.values() for d in s), (
            f"symbolic-shaped constant present (the bug): {consts}"
        )

        # The compile itself (the operation that fails today).
        torch._inductor.aoti_compile_and_package(ep)
