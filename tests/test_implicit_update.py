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

from neml2.equation_systems import ModelNonlinearSystem
from neml2.model import Model
from neml2.models.common import ImplicitUpdate
from neml2.solvers import Newton, RetCode
from neml2.types import Scalar


class ScalarResidual(Model):
    """Toy 1-D residual `x² − c = 0` used by the abstract ImplicitUpdate tests.

    Uses the C++-aligned residual name `<var>_residual`, so ModelNonlinearSystem
    can infer the residual name from the unknown name without an explicit
    `residuals = ...` override.
    """

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


def test_implicit_update_returns_converged_unknown():
    system = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    update = ImplicitUpdate(system, Newton(atol=1e-12, rtol=1e-12, miters=20))

    (x,) = update(
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([4.0, 9.0], dtype=torch.float64),
    )

    assert update.last_status is RetCode.SUCCESS
    # ImplicitUpdate.forward returns typed wrappers so
    # sub_batch_ndim propagates to downstream consumers without re-wrap drift.
    assert torch.allclose(x.data, torch.tensor([2.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_implicit_update_forward_v_uses_ift():
    system = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    update = ImplicitUpdate(system, Newton(atol=1e-12, rtol=1e-12, miters=20))
    x0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    c = torch.tensor([4.0, 9.0], dtype=torch.float64)
    Vc = Scalar(torch.ones(1, 2, dtype=torch.float64))

    x, v_out = update(x0, c, v={"c": {"c": Vc}})

    expected = 1.0 / (2.0 * x.data)
    assert torch.allclose(v_out["x"]["c"].data.squeeze(0), expected, atol=1e-12)


def test_implicit_update_input_output_specs():
    system = ModelNonlinearSystem(ScalarResidual(), unknowns=[["x"]])
    update = ImplicitUpdate(system, Newton())

    assert list(update.input_spec) == ["x", "c"]
    assert list(update.output_spec) == ["x"]
