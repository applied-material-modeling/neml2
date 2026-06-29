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

"""Smoke tests for the ported crystallographic-texture postprocessing."""

import math

import torch

import neml2.texture as tex
from neml2.types import MRP, Scalar, Vec

torch.manual_seed(0)


def test_kdeodf_probabilities_finite_positive():
    """KDEODF.forward yields finite, strictly positive probabilities."""
    odf = tex.KDEODF(MRP.rand(50), tex.DeLaValleePoussinKernel(torch.tensor(0.2)))
    probs = odf.forward(MRP.rand(10))
    assert probs.shape == (10,)
    assert torch.isfinite(probs).all()
    assert (probs > 0).all()


def test_symmetry_operators_count():
    """432 cubic group has 24 proper operators, 48 with inversion."""
    ops = tex.symmetry_operators_as_R2("432")
    assert ops.sub_batch_shape[0] == 24
    ops_inv = tex.symmetry_operators_as_R2("432", include_inversion=True)
    assert ops_inv.sub_batch_shape[0] == 48


def test_projection_round_trip():
    """Project then inverse recovers the (mirrored) hemisphere point.

    The v2 stereographic / Lambert convention projects from the south pole,
    so ``inverse(forward(v))`` recovers ``(x, y, -z)`` -- the same point on
    the opposite hemisphere. Negating z back recovers the original upper
    hemisphere point to machine precision.
    """
    v = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.6, 0.0, 0.8],
            [0.0, 0.8, 0.6],
            [0.3, 0.4, math.sqrt(1.0 - 0.09 - 0.16)],
        ],
        dtype=torch.float64,
    )
    for proj in (tex.StereographicProjection(), tex.LambertProjection()):
        back = proj.inverse(proj(v))
        # back == (x, y, -z); recover original by negating z
        recovered = back.clone()
        recovered[..., 2] *= -1.0
        assert torch.allclose(recovered, v, atol=1e-10)
        # results lie on the unit sphere (a hemisphere)
        assert torch.allclose(
            torch.linalg.norm(back, dim=-1),
            torch.ones(back.shape[0], dtype=torch.float64),
            atol=1e-10,
        )


def test_rotation_quadrature():
    """rotation_quadrature returns matched MRP points / Scalar weights, weights > 0."""
    points, weights = tex.rotation_quadrature(4)
    assert isinstance(points, MRP)
    assert isinstance(weights, Scalar)
    assert points.batch.shape == weights.batch.shape
    assert (weights.data > 0).all()


def test_pole_figure_points_runs():
    """pole_figure_points returns a finite (..., 2) point set without error."""
    pts = tex.pole_figure_points(MRP.rand(20), Vec(torch.tensor([0.0, 0.0, 1.0])))
    assert pts.shape[-1] == 2
    assert torch.isfinite(pts).all()
    # with a non-trivial crystal symmetry
    pts2 = tex.pole_figure_points(
        MRP.rand(20), Vec(torch.tensor([0.0, 0.0, 1.0])), crystal_symmetry="432"
    )
    assert pts2.shape[-1] == 2
    assert torch.isfinite(pts2).all()


def test_inverse_pole_figure_points_runs():
    """inverse_pole_figure_points returns a finite (..., 2) point set without error."""
    pts = tex.inverse_pole_figure_points(
        MRP.rand(20), Vec(torch.tensor([0.0, 0.0, 1.0])), crystal_symmetry="432"
    )
    assert pts.shape[-1] == 2
    assert torch.isfinite(pts).all()


def test_pole_figure_odf_matches_v2_reference():
    """Golden values pinned against the v2 ``postprocessing.pole_figure_odf``
    (built v2.1.6) for a fixed orientation set. This guards the fiber/symmetry
    composition: v2 wrote ``r1.rotate(r2)`` (= ``r2 * r1`` = ``compose(r2, r1)``);
    reversing the operands to ``compose(r1, r2)`` leaves the cube/identity case
    intact but washes out a general texture, so a smoke test misses it -- these
    values do not."""
    torch.set_default_dtype(torch.float64)
    X = MRP(
        torch.tensor(
            [
                [0.10, 0.20, -0.05],
                [-0.15, 0.05, 0.20],
                [0.02, -0.22, 0.11],
                [0.18, 0.04, 0.09],
                [-0.07, -0.12, 0.21],
                [0.25, -0.03, -0.10],
                [0.01, 0.17, 0.14],
                [-0.20, 0.09, -0.06],
            ]
        )
    )
    odf = tex.KDEODF(X, tex.DeLaValleePoussinKernel(torch.tensor(0.25)))
    odf.kernel.h = torch.tensor(0.25)
    vals, _ = tex.pole_figure_odf(
        odf, torch.tensor([1.0, 1.0, 1.0]), crystal_symmetry="432", nradial=3, ntheta=4, nquad=12
    )
    expected = torch.tensor(
        [
            [0.725414, 0.715360, 0.718828, 0.728918, 0.725414],
            [0.540768, 1.110577, 0.713709, 0.363413, 0.540768],
            [1.942825, 1.608087, 1.942825, 1.608087, 1.942825],
        ]
    )
    assert torch.allclose(vals, expected, atol=1e-5)


def test_pretty_plot_helpers_importable():
    """The matplotlib-backed helpers are importable (not called -- need a display)."""
    assert callable(tex.pretty_plot_pole_figure_odf)
    assert callable(tex.pretty_plot_pole_figure_points)
    assert callable(tex.pretty_plot_inverse_pole_figure)


def test_pretty_plots_render():
    """Drive the matplotlib rendering paths of all three pretty-plot helpers on
    the Agg backend, including the explicit-limits and Lambert-projection
    branches. Coarse grids keep it fast."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    torch.set_default_dtype(torch.float64)
    ori = MRP.rand(60)
    pole = torch.tensor([1.0, 1.0, 1.0])
    odf = tex.KDEODF(ori, tex.DeLaValleePoussinKernel(torch.tensor(0.2)))
    odf.kernel.h = torch.tensor(0.25)

    tex.pretty_plot_pole_figure_odf(
        odf, pole, crystal_symmetry="432", nradial=6, ntheta=8, nquad=12
    )
    tex.pretty_plot_pole_figure_odf(
        odf,
        pole,
        crystal_symmetry="432",
        nradial=6,
        ntheta=8,
        nquad=12,
        limits=(0.0, 3.0),
        ncontour=8,
    )
    tex.pretty_plot_pole_figure_odf(
        odf, pole, projection="lambert", crystal_symmetry="432", nradial=6, ntheta=8, nquad=12
    )
    tex.pretty_plot_pole_figure_points(ori, pole, crystal_symmetry="432")
    tex.pretty_plot_inverse_pole_figure(ori, torch.tensor([0.0, 1.0, 0.0]), crystal_symmetry="432")
    plt.close("all")


def test_optimize_kernel_runs():
    """The cross-validation kernel-width optimizer runs and yields a positive,
    finite half-width."""
    torch.set_default_dtype(torch.float64)
    odf = tex.KDEODF(MRP.rand(40), tex.DeLaValleePoussinKernel(torch.tensor(0.2)))
    odf.optimize_kernel(miter=3)
    h = odf.kernel.h.detach()
    assert torch.isfinite(h) and float(h) > 0


def test_texture_index_positive_finite():
    """texture_index integrates to a positive, finite scalar."""
    torch.set_default_dtype(torch.float64)
    odf = tex.KDEODF(MRP.rand(200), tex.DeLaValleePoussinKernel(torch.tensor(0.2)))
    odf.kernel.h = torch.tensor(0.3)
    J = float(odf.texture_index())
    assert math.isfinite(J) and J > 0


def test_split_leaves_one_out():
    """split(X, i) returns the other n-1 orientations and the i-th singleton."""
    torch.set_default_dtype(torch.float64)
    X = MRP.rand(5)
    keep, left = tex.split(X, 2)
    assert keep.batch.shape[0] == 4
    assert left.batch.ndim == 0


def test_spherical_and_gauss_quadrature():
    """gauss_points / spherical_quadrature return finite points and weights."""
    torch.set_default_dtype(torch.float64)
    pts, wts = tex.gauss_points(8)
    assert pts.shape == (8,) and wts.shape == (8,)
    assert torch.isfinite(pts).all() and (wts > 0).all()
    spts, swts = tex.spherical_quadrature(6)
    assert torch.isfinite(spts).all() and torch.isfinite(swts).all()


def test_cart2polar_round_trip():
    """cart2polar maps planar points to (theta, r); r recovers the radius."""
    torch.set_default_dtype(torch.float64)
    v = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
    polar = tex.cart2polar(v)
    assert torch.allclose(polar[:, 1], torch.linalg.norm(v, dim=-1))


def test_ipf_reduction_keeps_fundamental_subset():
    """IPFReduction selects the cubic fundamental wedge -- a subset of the input,
    and idempotent on its own output."""
    torch.set_default_dtype(torch.float64)
    reduction = tex.IPFReduction()  # default cubic 001/101/111 wedge
    v = Vec(torch.randn(200, 3))
    v = Vec(v.data / torch.linalg.norm(v.data, dim=-1, keepdim=True))
    kept = reduction(v)
    assert kept.batch.ndim == 1
    assert 0 < kept.batch.shape[0] <= 200
    # re-reducing keeps everything already inside the wedge
    assert reduction(kept).batch.shape[0] == kept.batch.shape[0]


def test_ipf_reduction_rejects_batched_limits():
    """The fundamental-region limits must be single directions."""
    torch.set_default_dtype(torch.float64)
    try:
        tex.IPFReduction(v0=Vec(torch.randn(3, 3)))
    except ValueError:
        return
    raise AssertionError("IPFReduction should reject batched limit directions")
