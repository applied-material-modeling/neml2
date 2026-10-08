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

r"""Eager input-sensitivity through a sub-batched :class:`ImplicitUpdate`.

The native ``forward(v=)`` branch extracts each variable block from the assembled
``du/dg`` and contracts the given tangent through it. A DENSE sub-batch group folds
its whole sub-batch into base, so the extraction width must be ``base * prod(sub)``
(not base alone) for the contraction to line up -- these tests pin that the fixed
width gives the correct per-site derivative. A BLOCK group keeps the common sub-batch
as a paired intermediate axis the eager pushforward can't fold back, so it is rejected
(mirroring the compiled route, which also rejects sub-batched implicit derivative pairs).
"""

from __future__ import annotations

import pytest
import torch

from neml2 import load_string
from neml2.types import Scalar


def _model(structure: str, weight_u: float, nb: int, n: int):
    """Linear scalar implicit system ``weight_u * u - g = 0`` with a sub-batch axis."""
    return load_string(
        f"""
        [Settings]
          [example_batch_shape]
            u = '({nb}; {n})'
            g = '({nb}; {n})'
          []
        []
        [Models]
          [residual]
            type = ScalarLinearCombination
            from = 'u g'
            to = 'r'
            weights = '{weight_u} -1'
          []
        []
        [EquationSystems]
          [eq_sys]
            type = NonlinearSystem
            model = 'residual'
            unknowns = 'u'
            residuals = 'r'
            structure = '{structure}'
          []
        []
        [Solvers]
          [lu] type = DenseLU []
          [newton]
            type = Newton
            linear_solver = 'lu'
            abs_tol = 1e-12
            rel_tol = 1e-10
            max_its = 25
          []
        []
        [Models]
          [model] type = ImplicitUpdate equation_system = 'eq_sys' solver = 'newton' []
        []
        """
    ).get_model("model")


def _inputs(nb: int, n: int, g_val: torch.Tensor):
    return {
        "u": Scalar(torch.zeros(nb, n, dtype=torch.float64), 1).with_sub_batch_ndim(1),
        "g": Scalar(g_val, 1).with_sub_batch_ndim(1),
    }


def test_dense_subbatch_sensitivity_matches_finite_difference():
    """DENSE sub-batch ``du/dg`` is correct per site: ``weight_u u = g`` -> ``du/dg = 1/weight_u``.

    Exercises the folded extraction width (``base * prod(sub)``) and the pushforward
    contraction over a non-trivial sub-batch, both numerically and against a forward
    finite difference.
    """
    nb, n, w = 2, 4, 2.0
    model = _model("dense", w, nb, n)
    torch.manual_seed(0)
    g = torch.randn(nb, n, dtype=torch.float64)
    tangent = {"g": Scalar(torch.ones(nb, n, dtype=torch.float64), 1).with_sub_batch_ndim(1)}
    _, jvps = model.jvp(_inputs(nb, n, g), tangent)
    analytic = jvps["u"].data.reshape(nb, n)
    torch.testing.assert_close(analytic, torch.full((nb, n), 1.0 / w, dtype=torch.float64))

    eps = 1e-6
    u_plus = model.call_by_name(_inputs(nb, n, g + eps))["u"].data.reshape(nb, n)
    u_minus = model.call_by_name(_inputs(nb, n, g - eps))["u"].data.reshape(nb, n)
    torch.testing.assert_close(analytic, (u_plus - u_minus) / (2 * eps), rtol=0, atol=1e-7)


def test_block_subbatch_sensitivity_is_rejected():
    """BLOCK sub-batch sensitivity raises -- parity with the compiled route's rejection."""
    nb, n = 2, 3
    model = _model("block", 1.0, nb, n)
    g = torch.randn(nb, n, dtype=torch.float64)
    tangent = {"g": Scalar(torch.ones(nb, n, dtype=torch.float64), 1).with_sub_batch_ndim(1)}
    with pytest.raises(NotImplementedError, match="BLOCK-structured"):
        model.jvp(_inputs(nb, n, g), tangent)
