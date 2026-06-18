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

"""``neml2::opaque_pow`` must stay an opaque Inductor fusion barrier under AOTI.

``opaque_pow`` exists so the expensive ``pow`` in rate-dependent leaves
(``PowerLawSlipRule``, ``PerzynaPlasticFlowRate``) is computed once per
``(B, n_comp)`` rather than being inlined into the downstream K-batched
per-component reduction and recomputed ``K x n_comp`` times. The barrier only
helps if the op stays *opaque* through the AOTI lowering path
(``torch.export`` + ``run_decompositions()``).

This is fragile against the op's registration: a bare ``lib.impl(..., "Meta")``
or registering the forward on the ``Autograd`` key both make
``run_decompositions()`` inline the op back to ``aten.pow``, silently
dissolving the barrier (it stays opaque in eager / ``torch.compile``, so only
the AOTI path regresses -- a ~2x slowdown on the crystal-plasticity suite).
The fix is ``register_fake`` + ``register_autograd``; these tests pin it.
"""

from __future__ import annotations

import torch

import neml2.types.functions  # noqa: F401  -- registers neml2::opaque_pow


class _Pow(torch.nn.Module):
    def forward(self, x):
        return torch.ops.neml2.opaque_pow(x, torch.tensor(7.0, dtype=x.dtype))


def _post_decomposition_ops() -> list[str]:
    x = torch.rand(8, dtype=torch.float64)
    ep = torch.export.export(_Pow(), (x,)).run_decompositions()
    return [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]


def test_opaque_pow_survives_aoti_decomposition():
    """The op stays opaque through ``run_decompositions()`` (the AOTI path) --
    it must NOT be inlined back to ``aten.pow``."""
    ops = _post_decomposition_ops()
    assert any("opaque_pow" in op for op in ops), (
        f"opaque_pow was decomposed away (barrier lost): {ops}"
    )
    assert not any(op == "aten.pow.Tensor_Tensor" for op in ops), (
        f"a bare aten.pow leaked into the lowered graph: {ops}"
    )


def test_opaque_pow_value_matches_torch_pow():
    x = torch.rand(16, dtype=torch.float64) + 0.1
    e = torch.tensor(3.5, dtype=torch.float64)
    assert torch.allclose(torch.ops.neml2.opaque_pow(x, e), torch.pow(x, e))


def test_opaque_pow_autograd_matches_torch_pow():
    """Eager autograd through the op (pyzag adjoint / in-process compile path)
    must match ``torch.pow`` for BOTH the base and the exponent."""
    # Build the base as a leaf AFTER the +0.1 shift (positive base so the
    # exponent gradient's log(base) is finite).
    base = (torch.rand(5, dtype=torch.float64) + 0.1).requires_grad_(True)
    expo = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    torch.ops.neml2.opaque_pow(base, expo).sum().backward()
    assert base.grad is not None and expo.grad is not None
    g_base, g_expo = base.grad, expo.grad

    base_ref = base.detach().clone().requires_grad_(True)
    expo_ref = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    torch.pow(base_ref, expo_ref).sum().backward()
    assert base_ref.grad is not None and expo_ref.grad is not None

    assert torch.allclose(g_base, base_ref.grad)
    assert torch.allclose(g_expo, expo_ref.grad)
