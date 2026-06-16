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

from __future__ import annotations

import torch

from neml2.es import (
    AssembledMatrix,
    AssembledVector,
    AxisLayout,
    ModelNonlinearSystem,
    SparseVector,
)
from neml2.models.model import Model
from neml2.solvers import DenseLU, Newton, NewtonWithLineSearch, RetCode
from neml2.types import Scalar, Tensor


class ScalarResidual(Model):
    input_spec = {"x": Scalar, "c": Scalar}
    output_spec = {"x_residual": Scalar}

    def forward(self, x: Scalar, c: Scalar, v=None):
        r = x * x - c
        if v is None:
            return r
        return r, self.apply_chain_rule(
            v,
            "x_residual",
            {
                "x": lambda V: 2.0 * x * V,
                "c": lambda V: -V,
            },
            output=r,
        )


def test_dense_lu_solves_batched_vector_rhs():
    layout = AxisLayout([["x"]], {"x": Scalar})
    A = AssembledMatrix(
        layout,
        layout,
        [[Tensor(torch.tensor([[[2.0]], [[4.0]]], dtype=torch.float64), batch_ndim=1)]],
    )
    b = AssembledVector(
        layout,
        [Tensor(torch.tensor([[6.0], [20.0]], dtype=torch.float64), batch_ndim=1)],
    )

    x = DenseLU().solve(A, b)

    assert isinstance(x, AssembledVector)
    assert torch.equal(x.tensors[0].data, torch.tensor([[3.0], [5.0]], dtype=torch.float64))


def test_dense_lu_solves_batched_matrix_rhs():
    layout = AxisLayout([["x"]], {"x": Scalar})
    rhs_layout = AxisLayout([["c"]], {"c": Scalar})
    A = AssembledMatrix(
        layout,
        layout,
        [[Tensor(torch.tensor([[[2.0]], [[4.0]]], dtype=torch.float64), batch_ndim=1)]],
    )
    B = AssembledMatrix(
        layout,
        rhs_layout,
        [[Tensor(torch.tensor([[[6.0]], [[20.0]]], dtype=torch.float64), batch_ndim=1)]],
    )

    X = DenseLU().solve(A, B)

    assert isinstance(X, AssembledMatrix)
    assert torch.equal(X.tensors[0][0].data, torch.tensor([[[3.0]], [[5.0]]], dtype=torch.float64))


def test_newton_converges_batched_scalar_system():
    sys = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    sys.initialize(
        u=SparseVector(sys.ulayout, {"x": Scalar(torch.tensor([1.0, 2.0], dtype=torch.float64))}),
        g=SparseVector(sys.glayout, {"c": Scalar(torch.tensor([4.0, 9.0], dtype=torch.float64))}),
    )

    result = Newton(atol=1e-12, rtol=1e-12, miters=20).solve(sys)
    solved = sys.u().disassemble()["x"]

    assert result.ret is RetCode.SUCCESS
    from neml2.types import allclose as _allclose

    assert _allclose(solved, torch.tensor([2.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_newton_with_linesearch_converges_same_as_newton():
    """Linesearch should converge the well-behaved x^2 - c = 0 case identically."""
    sys = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    sys.initialize(
        u=SparseVector(sys.ulayout, {"x": Scalar(torch.tensor([1.0, 2.0], dtype=torch.float64))}),
        g=SparseVector(sys.glayout, {"c": Scalar(torch.tensor([4.0, 9.0], dtype=torch.float64))}),
    )

    result = NewtonWithLineSearch(atol=1e-12, rtol=1e-12, miters=20).solve(sys)
    solved = sys.u().disassemble()["x"]

    assert result.ret is RetCode.SUCCESS
    from neml2.types import allclose as _allclose

    assert _allclose(solved, torch.tensor([2.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_newton_with_linesearch_strong_wolfe_variant():
    """The STRONG_WOLFE step rule should also converge the same problem."""
    sys = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    sys.initialize(
        u=SparseVector(sys.ulayout, {"x": Scalar(torch.tensor([1.0, 2.0], dtype=torch.float64))}),
        g=SparseVector(sys.glayout, {"c": Scalar(torch.tensor([4.0, 9.0], dtype=torch.float64))}),
    )

    result = NewtonWithLineSearch(
        atol=1e-12, rtol=1e-12, miters=20, linesearch_type="STRONG_WOLFE"
    ).solve(sys)
    solved = sys.u().disassemble()["x"]
    assert result.ret is RetCode.SUCCESS
    from neml2.types import allclose as _allclose

    assert _allclose(solved, torch.tensor([2.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_newton_with_linesearch_rejects_bad_linesearch_type():
    import pytest

    with pytest.raises(ValueError, match="linesearch_type"):
        NewtonWithLineSearch(linesearch_type="HERMES")
