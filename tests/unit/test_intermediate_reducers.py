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

"""Regression coverage for the intermediate reducers after dropping ``retag(1)``."""

from __future__ import annotations

import pytest
import torch

from neml2 import load_string
from neml2.types import SR2, Scalar


def _model(typ: str):
    return load_string(
        f"""
        [Models]
          [m]
            type = {typ}
            from = 'x'
            to = 'y'
          []
        []
        """
    ).get_model("m")


def test_scalar_sum_preserves_common_prefix():
    x = Scalar(torch.arange(1, 13, dtype=torch.float64).reshape(2, 2, 3), 1).with_sub_batch_ndim(2)
    model = _model("ScalarIntermediateSum")
    out = model.call_by_name({"x": x})["y"]
    assert out.sub_batch_ndim == 1
    assert tuple(out.data.shape) == (2, 2)
    torch.testing.assert_close(out.data, x.data.sum(dim=-1))


def test_sr2_sum_preserves_common_prefix():
    data = torch.arange(0, 2 * 3 * 4 * 6, dtype=torch.float64).reshape(2, 3, 4, 6)
    x = SR2(data, 3).with_sub_batch_ndim(2)
    model = _model("SR2IntermediateSum")
    out = model.call_by_name({"x": x})["y"]
    assert out.sub_batch_ndim == 1
    assert tuple(out.data.shape) == (2, 3, 6)
    torch.testing.assert_close(out.data, x.data.sum(dim=-2))


def test_sr2_mean_preserves_common_prefix():
    data = torch.arange(0, 2 * 3 * 4 * 6, dtype=torch.float64).reshape(2, 3, 4, 6)
    x = SR2(data, 3).with_sub_batch_ndim(2)
    model = _model("SR2IntermediateMean")
    out = model.call_by_name({"x": x})["y"]
    assert out.sub_batch_ndim == 1
    assert tuple(out.data.shape) == (2, 3, 6)
    torch.testing.assert_close(out.data, x.data.mean(dim=-2))


@pytest.mark.parametrize(
    "typ",
    ["ScalarIntermediateSum", "SR2IntermediateSum", "SR2IntermediateMean"],
)
def test_zero_sub_batch_raises(typ):
    if typ.startswith("Scalar"):
        x = Scalar(torch.tensor([1.0, 2.0, 3.0]))
    else:
        x = SR2(torch.arange(1, 7, dtype=torch.float64), 0)
    model = _model(typ)
    with pytest.raises(ValueError, match="requires at least one sub-batch dim"):
        model.call_by_name({"x": x})


def test_jvp_scalar_sum():
    x = Scalar(torch.arange(1, 13, dtype=torch.float64).reshape(2, 2, 3), 1).with_sub_batch_ndim(2)
    tangent = Scalar(torch.ones(2, 2, 3, dtype=torch.float64), 1).with_sub_batch_ndim(2)
    model = _model("ScalarIntermediateSum")
    _, jvps = model.jvp({"x": x}, {"x": tangent})
    y = jvps["y"]
    assert y.sub_batch_ndim == 1
    assert tuple(y.data.shape) == (2, 2)
    torch.testing.assert_close(y.data, torch.full((2, 2), 3.0))


def test_jvp_sr2_sum():
    data = torch.arange(0, 2 * 3 * 4 * 6, dtype=torch.float64).reshape(2, 3, 4, 6)
    x = SR2(data, 3).with_sub_batch_ndim(2)
    tangent = SR2(torch.ones(2, 3, 4, 6, dtype=torch.float64), 3).with_sub_batch_ndim(2)
    model = _model("SR2IntermediateSum")
    _, jvps = model.jvp({"x": x}, {"x": tangent})
    y = jvps["y"]
    assert y.sub_batch_ndim == 1
    assert tuple(y.data.shape) == (2, 3, 6)
    torch.testing.assert_close(y.data, torch.full((2, 3, 6), 4.0))


def test_jvp_sr2_mean():
    data = torch.arange(0, 2 * 3 * 4 * 6, dtype=torch.float64).reshape(2, 3, 4, 6)
    x = SR2(data, 3).with_sub_batch_ndim(2)
    tangent = SR2(torch.ones(2, 3, 4, 6, dtype=torch.float64), 3).with_sub_batch_ndim(2)
    model = _model("SR2IntermediateMean")
    _, jvps = model.jvp({"x": x}, {"x": tangent})
    y = jvps["y"]
    assert y.sub_batch_ndim == 1
    assert tuple(y.data.shape) == (2, 3, 6)
    torch.testing.assert_close(y.data, torch.full((2, 3, 6), 1.0))


def test_scalar_intermediate_mean_forward_and_zero_raises():
    x = Scalar(torch.arange(1, 7, dtype=torch.float64).reshape(3, 2), 1).with_sub_batch_ndim(1)
    model = _model("ScalarIntermediateMean")
    out = model.call_by_name({"x": x})["y"]
    assert out.sub_batch_ndim == 0
    # Scalar(base=1): data (3, 2) with sub_batch_ndim=1 -> sub is trailing (2,)
    # Mean over sub axis leaves dynamic batch (3,)
    assert tuple(out.data.shape) == (3,)
    torch.testing.assert_close(out.data, x.data.mean(dim=-1))

    # Zero sub-batch must raise.
    with pytest.raises(ValueError, match="requires at least one sub-batch dim"):
        model.call_by_name({"x": Scalar(torch.tensor(1.0))})


def test_scalar_sum_dim_0():
    # (student=3, class=2) -> sum over student (dim=0) leaves class=2
    x = Scalar(torch.arange(1, 7, dtype=torch.float64).reshape(3, 2), 1).with_sub_batch_ndim(2)
    model = load_string(
        """
        [Models]
          [m]
            type = ScalarIntermediateSum
            from = 'x'
            to = 'y'
            dim = 0
          []
        []
        """
    ).get_model("m")
    out = model.call_by_name({"x": x})["y"]
    assert out.sub_batch_ndim == 1
    assert tuple(out.data.shape) == (2,)
    torch.testing.assert_close(out.data, x.data.sum(dim=0))


# ---------------------------------------------------------------------------
# Cross-site Jacobian coverage checked against finite differences
#
# Note on API design & guarantees:
# The forward-mode chain rule (forward(v=) and Model.jvp) fully supports
# cross-site reductions across sub-batch dimensions by propagating directional
# tangents through action(V) = mean/sum(V.sub_batch, dim).
#
# In contrast, Model.jacobian is designed for evaluating pointwise base
# Jacobians on plain-batch inputs (*batch, *sub_batch, *out_base, *in_base)
# and does not assemble cross-site block matrices between distinct sub-batch spaces.
# ---------------------------------------------------------------------------


def test_cross_site_jacobian_sr2_mean_vs_finite_difference():
    # Mean over the only sub-batch axis (sub=2, base=6) -> output (6,)
    x_data = torch.arange(0, 12, dtype=torch.float64).reshape(2, 6)
    x = SR2(x_data, 3).with_sub_batch_ndim(1)
    model = _model("SR2IntermediateMean")

    eps = 1e-7
    J_fd = torch.zeros(6, 2, 6, dtype=torch.float64)  # (out_base, sub_in, in_base)
    for s in range(2):
        for b in range(6):
            xp = x_data.clone()
            xp[s, b] += eps
            xm = x_data.clone()
            xm[s, b] -= eps
            yp = model(SR2(xp, 3).with_sub_batch_ndim(1)).data
            ym = model(SR2(xm, 3).with_sub_batch_ndim(1)).data
            J_fd[:, s, b] = (yp - ym) / (2 * eps)

    J_jvp = torch.zeros(6, 2, 6, dtype=torch.float64)
    for s in range(2):
        for b in range(6):
            t = torch.zeros(2, 6, dtype=torch.float64)
            t[s, b] = 1.0
            tangent = SR2(t, 3).with_sub_batch_ndim(1)
            _, jvps = model.jvp({"x": x}, {"x": tangent})
            J_jvp[:, s, b] = jvps["y"].data

    torch.testing.assert_close(J_jvp, J_fd, rtol=1e-6, atol=1e-8)
    # Check expected coefficient: 1 / 2 sub-sites = 0.5 per site on diagonal base
    for s in range(2):
        torch.testing.assert_close(J_jvp[:, s, :], 0.5 * torch.eye(6, dtype=torch.float64))


def test_cross_site_jacobian_sr2_sum_vs_finite_difference():
    x_data = torch.arange(0, 12, dtype=torch.float64).reshape(2, 6)
    x = SR2(x_data, 3).with_sub_batch_ndim(1)
    model = _model("SR2IntermediateSum")

    eps = 1e-7
    J_fd = torch.zeros(6, 2, 6, dtype=torch.float64)
    for s in range(2):
        for b in range(6):
            xp = x_data.clone()
            xp[s, b] += eps
            xm = x_data.clone()
            xm[s, b] -= eps
            yp = model(SR2(xp, 3).with_sub_batch_ndim(1)).data
            ym = model(SR2(xm, 3).with_sub_batch_ndim(1)).data
            J_fd[:, s, b] = (yp - ym) / (2 * eps)

    J_jvp = torch.zeros(6, 2, 6, dtype=torch.float64)
    for s in range(2):
        for b in range(6):
            t = torch.zeros(2, 6, dtype=torch.float64)
            t[s, b] = 1.0
            tangent = SR2(t, 3).with_sub_batch_ndim(1)
            _, jvps = model.jvp({"x": x}, {"x": tangent})
            J_jvp[:, s, b] = jvps["y"].data

    torch.testing.assert_close(J_jvp, J_fd, rtol=1e-6, atol=1e-8)
    for s in range(2):
        torch.testing.assert_close(J_jvp[:, s, :], torch.eye(6, dtype=torch.float64))


def test_cross_site_jacobian_scalar_sum_dim_0_vs_finite_difference():
    # Multi-sub-batch: (student=3, class=2) -> sum over student (dim=0) leaves class=2
    x_data = torch.arange(1, 7, dtype=torch.float64).reshape(3, 2)
    x = Scalar(x_data, 1).with_sub_batch_ndim(2)
    model = load_string(
        """
        [Models]
          [m]
            type = ScalarIntermediateSum
            from = 'x'
            to = 'y'
            dim = 0
          []
        []
        """
    ).get_model("m")

    eps = 1e-7
    J_fd = torch.zeros(2, 3, 2, dtype=torch.float64)  # (out_sub, in_sub0, in_sub1)
    for i in range(3):
        for j in range(2):
            xp = x_data.clone()
            xp[i, j] += eps
            xm = x_data.clone()
            xm[i, j] -= eps
            yp = model(Scalar(xp, 1).with_sub_batch_ndim(2)).data
            ym = model(Scalar(xm, 1).with_sub_batch_ndim(2)).data
            J_fd[:, i, j] = (yp - ym) / (2 * eps)

    J_jvp = torch.zeros(2, 3, 2, dtype=torch.float64)
    for i in range(3):
        for j in range(2):
            t = torch.zeros(3, 2, dtype=torch.float64)
            t[i, j] = 1.0
            tangent = Scalar(t, 1).with_sub_batch_ndim(2)
            _, jvps = model.jvp({"x": x}, {"x": tangent})
            J_jvp[:, i, j] = jvps["y"].data

    torch.testing.assert_close(J_jvp, J_fd, rtol=1e-6, atol=1e-8)


def test_cross_site_jacobian_sr2_diff_vs_finite_difference():
    # Diff over trailing sub axis (dim=-1, n=1) -> sub preserved, length -1
    data = torch.arange(0, 2 * 3 * 4 * 6, dtype=torch.float64).reshape(2, 3, 4, 6)
    x = SR2(data, 3).with_sub_batch_ndim(2)
    model = load_string(
        """
        [Models]
          [m]
            type = SR2IntermediateDiff
            from = 'x'
            to = 'y'
            dim = -1
            n = 1
          []
        []
        """
    ).get_model("m")

    out = model(x)
    assert out.sub_batch_ndim == 2
    assert tuple(out.data.shape) == (2, 3, 3, 6)

    eps = 1e-7
    for s0 in range(3):
        for s1 in range(4):
            for b in [0, 2, 5]:
                xp = data.clone()
                xp[0, s0, s1, b] += eps
                xm = data.clone()
                xm[0, s0, s1, b] -= eps
                yp = model(SR2(xp, 3).with_sub_batch_ndim(2)).data
                ym = model(SR2(xm, 3).with_sub_batch_ndim(2)).data
                fd_col = (yp - ym) / (2 * eps)

                t = torch.zeros_like(data)
                t[0, s0, s1, b] = 1.0
                tangent = SR2(t, 3).with_sub_batch_ndim(2)
                _, jvps = model.jvp({"x": x}, {"x": tangent})
                jvp_col = jvps["y"].data
                torch.testing.assert_close(jvp_col, fd_col, rtol=1e-6, atol=1e-8)


def test_composed_model_intermediate_mean_chain_rule():
    """Verify that tangent propagation survives a full ComposedModel graph
    surrounding IntermediateMean (pre-scale -> intermediate mean -> post-scale).
    """
    factory = load_string(
        """
        [Models]
          [pre]
            type = SR2LinearCombination
            from = 'x'
            to = 's'
            weights = '2.0'
          []
          [reduce]
            type = SR2IntermediateMean
            from = 's'
            to = 's_mean'
            dim = -1
          []
          [post]
            type = SR2LinearCombination
            from = 's_mean'
            to = 'y'
            weights = '3.0'
          []
          [composed]
            type = ComposedModel
            models = 'pre reduce post'
          []
        []
        """
    )
    model = factory.get_model("composed")

    x_data = torch.arange(0, 24, dtype=torch.float64).reshape(4, 6)
    x = SR2(x_data, 3).with_sub_batch_ndim(1)

    out = model(x)
    out_val = out[0] if isinstance(out, tuple) else out
    assert out_val.sub_batch_ndim == 0
    assert tuple(out_val.data.shape) == (6,)

    # Forward value: 2 * (1/4) * 3 = 1.5 of sum(x)
    expected = 1.5 * x_data.sum(dim=0)
    torch.testing.assert_close(out_val.data, expected)

    # Tangent propagation through the composed graph
    for s in range(4):
        for b in range(6):
            t_data = torch.zeros(4, 6, dtype=torch.float64)
            t_data[s, b] = 1.0
            tangent = SR2(t_data, 3).with_sub_batch_ndim(1)
            _, jvps = model.jvp({"x": x}, {"x": tangent})
            expected_jvp = torch.zeros(6, dtype=torch.float64)
            expected_jvp[b] = 1.5
            torch.testing.assert_close(jvps["y"].data, expected_jvp)
