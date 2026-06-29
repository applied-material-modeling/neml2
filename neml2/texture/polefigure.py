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

"""Pole figures and inverse pole figures for crystallographic texture.

Ported from v2 ``neml2.postprocessing.polefigure``. Orientations are carried as
:class:`~neml2.types.MRP` (the v2 ``Rot``), directions as
:class:`~neml2.types.Vec`, and symmetry operators as :class:`~neml2.types.R2`
rotation matrices; orientation-space geometry stays typed via the free functions
:func:`~neml2.types.rotate` / :func:`~neml2.types.euler_rodrigues` /
:func:`~neml2.types.cross` / :func:`~neml2.types.vdot` / :func:`~neml2.types.unit`
and :func:`neml2.ops.symmetry`. The raw-tensor boundaries are the
``scipy.spatial.SphericalVoronoi`` quadrature, the Stereographic/Lambert
projection math (which operates on raw ``(..., 3)`` / ``(..., 2)`` arrays by
design), the upper-hemisphere / fundamental-region filters (ragged reductions),
and ``matplotlib`` plotting.
"""

from __future__ import annotations

from functools import reduce
from math import sqrt

import scipy.spatial as ss
import torch

from neml2.ops import symmetry
from neml2.types import (
    MRP,
    Scalar,
    Vec,
    cat,
    compose,
    cross,
    euler_rodrigues,
    rotate,
    unit,
    vdot,
)

_CPU = torch.device("cpu")


def arbitrary_hemispherical_quadrature(points, rtol=1e-5, atol=1e-8):
    """Calculate the weights for a spherical quadrature rule on the upper hemisphere

    Args:
        points (torch.tensor): tensor of shape (..., 3) of cartesian points on the upper hemisphere

    Keyword Args:
        rtol (float): relative tolerance for determining if points are on the equator
        atol (float): absolute tolerance for determining if points are on the equator

    Returns:
        torch.tensor: tensor of shape (...) giving the weights
    """
    orig_shape = points.shape[:-1]
    points = points.flatten(end_dim=-2)
    on_equator = torch.isclose(
        points[:, 2], torch.tensor(0.0, device=points.device), rtol=rtol, atol=atol
    )
    not_equator = torch.logical_not(on_equator)

    hemi_points = points[not_equator].clone()
    hemi_points[..., 2] *= -1

    augmented_points = torch.cat([points, hemi_points], dim=0)

    triangulation = ss.SphericalVoronoi(augmented_points.cpu().numpy())
    areas = torch.tensor(triangulation.calculate_areas(), device=points.device)
    areas = areas[: len(points)]
    areas[on_equator] *= 0.5

    return areas.reshape(orig_shape)


class StereographicProjection:
    """Stereographic projection"""

    def __init__(self):
        self.limit = 1.0

    def __call__(self, v):
        """Project from 3d points on sphere to 2d

        Args:
            v (torch.tensor): tensor of shape (..., 3)
        """
        return torch.stack([v[..., 0] / (1.0 + v[..., 2]), v[..., 1] / (1.0 + v[..., 2])], dim=-1)

    def inverse(self, v):
        """Inverse of the stereographic projection

        Args:
            v (torch.tensor): tensor of shape (..., 2)
        """
        X = v[..., 0]
        Y = v[..., 1]
        bot = 1.0 + X**2 + Y**2
        return torch.stack(
            [
                2.0 * X / bot,
                2.0 * Y / bot,
                (X**2 + Y**2 - 1.0) / bot,
            ],
            dim=-1,
        )


class LambertProjection:
    """Lambert equal area projection"""

    def __init__(self):
        self.limit = sqrt(2.0)

    def __call__(self, v):
        """Project from 3d points on sphere to 2d

        Args:
            v (torch.tensor): tensor of shape (..., 3)
        """
        return torch.stack(
            [
                torch.sqrt(2.0 / (1.0 + v[..., 2])) * v[..., 0],
                torch.sqrt(2.0 / (1.0 + v[..., 2])) * v[..., 1],
            ],
            dim=-1,
        )

    def inverse(self, v):
        """Inverse of the Lambert projection
        Args:
            v (torch.tensor): tensor of shape (..., 2)
        """
        X = v[..., 0]
        Y = v[..., 1]
        f = torch.sqrt(1.0 - (X**2 + Y**2) / 4.0)
        return torch.stack(
            [
                f * X,
                f * Y,
                -1.0 + (X**2 + Y**2) / 2.0,
            ],
            dim=-1,
        )


available_projections = {
    "stereographic": StereographicProjection(),
    "lambert": LambertProjection(),
}


def symmetry_operators_as_R2(orbifold, device=_CPU, include_inversion=False):
    """Return the symmetry operators for a given symmetry group as a batch of rank two tensors

    Args:
        orbifold (str): symmetry group in orbifold notation

    Keyword Args:
        device (torch.device): which device to place the tensors
        include_inversion (bool): whether to include inversion in the symmetry
            operators, default False

    Returns:
        neml2.types.R2: an R2 whose single sub-batch axis indexes the operators
    """
    ops = symmetry(orbifold, device=device)
    if include_inversion:
        # Concatenate the operators and their negatives along the operator
        # (sub-batch) axis, staying typed.
        return cat([ops.sub_batch, (-ops).sub_batch], dim=0)
    return ops


def pole_figure_odf(
    odf,
    pole,
    projection="stereographic",
    crystal_symmetry="1",
    nradial=20,
    ntheta=40,
    nquad=50,
    nchunk=1000,
    offset_fraction=1e-2,
):
    """Project an ODF onto a pole figure

    Args:
        odf (neml2.texture.ODF): ODF to project
        pole (torch.tensor or neml2.types.Vec): pole to project

    Keyword Args:
        projection (str): which polar projection to use, options are
            "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the
            crystal symmetry to apply to the base orientations, default "1"
        nradial (int): number of radial points to use
        ntheta (int): number of angular points to use
        nquad (int): number of points to integrate the pole figure
        nchunk (int): subbatch size to use in evaluating odf
        offset_fraction (float): fraction of the radial range to offset the
            pole figure to avoid singularities
    Returns:
        (values, mesh): tuple of the projected values and the (theta, R) mesh
            points used to make a contour plot
    """
    # Do some setup
    if not isinstance(pole, Vec):
        pole = Vec(pole)
    pole = unit(pole)
    projection = available_projections[projection]

    # Make the grid
    r = torch.linspace(
        projection.limit / nradial * offset_fraction, projection.limit, nradial, device=pole.device
    )
    theta_total = torch.linspace(0, 2 * torch.pi, ntheta + 1, device=pole.device)
    theta = theta_total[:-1]
    R, T = torch.meshgrid(r, theta, indexing="ij")

    # Get points on plane
    X = R * torch.cos(T)
    Y = R * torch.sin(T)

    # Project back to sphere
    sample_poles = Vec(projection.inverse(torch.stack([X, Y], dim=-1)))

    # Get all the equivalent poles. ``symmetry`` returns the operators on a
    # sub-batch axis; v2 carried them on the dynamic batch, so drop the sub-batch
    # declaration to expose the operator axis as a plain dynamic-batch axis.
    symmetry_operators = symmetry_operators_as_R2(crystal_symmetry, device=pole.device)
    crystal_poles = rotate(pole, symmetry_operators).with_sub_batch_ndim(0)
    crystal_poles = crystal_poles.dynamic_batch.unsqueeze(1).dynamic_batch.unsqueeze(1)

    # Calculate the static rotation from crystal to sample poles. ``rotation_from_to``
    # broadcasts only over equal-ndim operands, so lift the (nradial, ntheta) sample
    # grid to (1, nradial, ntheta) to align with crystal_poles' (n_ops, 1, 1).
    sample_poles_b = sample_poles.dynamic_batch.unsqueeze(0)
    r1 = MRP.rotation_from_to(crystal_poles, sample_poles_b)

    # Calculate the rotations about the sample poles
    # Note my diabolical cleverness -- we know the arc length metric is constant in
    # the standard Rodrigues space, so we can use a special function to that gets
    # standard rodrigues parameters for the rotation about the pole and convert to MRP
    # and then drop the metric term.
    angle_values = torch.linspace(0, 2 * torch.pi, nquad + 1, device=pole.device)
    dphi = torch.diff(angle_values)
    phi = Scalar(angle_values[:-1]).dynamic_batch.unsqueeze(1).dynamic_batch.unsqueeze(1)
    r2 = MRP.from_axis_angle_standard(sample_poles_b, phi)

    # Compose the two rotations (r1 ∘ r2) -- orientation composition, not a
    # matrix-vector rotation. v2's ``Rot.rotate(Rot)`` was composition.
    rm = compose(r1, r2.dynamic_batch.unsqueeze(1))

    # Calculate the ODF values. Flatten the dynamic-batch grid to a single axis
    # (raw reshape at the ODF-eval boundary), chunk through the ODF, reshape back.
    original_shape = rm.dynamic_batch.shape
    rm = rm.data.reshape(-1, 3)  # data-ok texture
    A = torch.cat(
        [odf(MRP(rm[i : i + nchunk])) for i in range(0, rm.shape[0], nchunk)], dim=0
    ).reshape(original_shape)

    # Incremental rotation
    dl = (dphi).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Calculate the actual pole values
    vals = torch.mean(torch.sum(A * dl, dim=0), dim=0) / (2 * torch.pi)

    # Calculate the normalization to get MRD
    weights = arbitrary_hemispherical_quadrature(sample_poles.data)  # data-ok texture
    nf = torch.sum(weights * vals) / (2 * torch.pi)
    vals = vals / nf

    # Make the mesh that matplotlib wants
    r[0] = 0.0
    TR, TT = torch.meshgrid(r, theta_total, indexing="ij")

    Tvals = torch.zeros_like(TR)
    Tvals[..., :-1] = vals
    Tvals[..., -1] = vals[..., 0]

    return Tvals, (TT, TR)


def pretty_plot_pole_figure_odf(
    odf,
    pole,
    projection="stereographic",
    crystal_symmetry="1",
    nradial=20,
    ntheta=40,
    nquad=50,
    nchunk=1000,
    limits=None,
    ncontour=20,
):
    """Project an ODF onto a pole figure

    Args:
        odf (neml2.texture.ODF): ODF to project
        pole (torch.tensor or neml2.types.Vec): pole to project

    Keyword Args:
        projection (str): which polar projection to use, options are
            "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the
            crystal symmetry to apply to the base orientations, default "1"
        nradial (int): number of radial points to use
        ntheta (int): number of angular points to use
        nquad (int): number of points to integrate the pole figure
        nchunk (int): subbatch size to use in evaluating odf
        limits ((float,float)): (vmin, vmax) to pass to contourf
        ncontour (int): number of contours to use, only used if limits is
            provided
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import NullLocator  # noqa: PLC0415

    vals, mesh = pole_figure_odf(
        odf, pole, projection, crystal_symmetry, nradial, ntheta, nquad, nchunk
    )
    projection = available_projections[projection]

    # Plot
    plt.figure()
    ax = plt.subplot(111, projection="polar")
    if limits is not None:
        CS = ax.contourf(
            mesh[0].cpu(),
            mesh[1].cpu(),
            vals.detach().cpu(),
            cmap="Greys",
            levels=torch.linspace(limits[0], limits[1], ncontour),
            extend="both",
        )
    else:
        CS = ax.contourf(mesh[0], mesh[1], vals.detach(), cmap="Greys")

    # Make the graph nice
    plt.ylim([0, projection.limit])
    ax.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.xaxis.set_minor_locator(NullLocator())
    plt.colorbar(CS, label="MRD")


def pole_figure_points(
    orientations,
    pole,
    projection="stereographic",
    crystal_symmetry="1",
    sample_symmetry="1",
):
    """Project crystal orientations to points on a pole figure

    Args:
        orientations (torch.tensor or neml2.types.MRP): tensor of orientations
            as *modified* Rodrigues parameters in the *active* convention with
            arbitrary batch shape.
        pole (torch.tensor or neml2.types.Vec): pole to project, must
            broadcast with orientations

    Keyword Args:
        projection (str): which polar projection to use, options are
            "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the
            crystal symmetry to apply to the base orientations, default "1"
        sample_symmetry (str): string giving the orbifold notation for the
            sample symmetry to apply to the projected points, default "1"

    Returns:
        torch.tensor: tensor with same batch shape as orientations but with end
            dimension (2,) giving each point
    """
    # Do some setup
    if not isinstance(pole, Vec):
        pole = Vec(pole)
    pole = unit(pole)
    if not isinstance(orientations, MRP):
        orientations = MRP(orientations)

    # Get all the equivalent poles
    symmetry_operators = symmetry_operators_as_R2(crystal_symmetry, device=orientations.device)
    equivalent_poles = rotate(pole, symmetry_operators)

    # Move from crystal to sample
    Rmat = euler_rodrigues(orientations)
    sample_poles = rotate(equivalent_poles, Rmat.dynamic_batch.unsqueeze(-1))
    # Apply sample symmetry
    sample_symmetry_operators = symmetry_operators_as_R2(
        sample_symmetry, device=orientations.device
    )
    sample_poles = rotate(sample_poles, sample_symmetry_operators.sub_batch.unsqueeze(0))

    # For my reference, at this point we have a tensor of shape
    # (arbitrary_batch_shape,) + (crystal_symmetry,) + (sample_symmetry,) + (3,)
    sample_poles = sample_poles.data  # data-ok texture

    # Eliminate poles on the lower hemisphere
    sample_poles = sample_poles[sample_poles[..., 2] >= 0.0]

    # Project and return
    projection = available_projections[projection]
    return projection(sample_poles)


def cart2polar(v):
    """Convert cartesian 2D points to polar coordinates

    Args:
        v (torch.tensor): tensor of shape (...,2)
    """
    return torch.stack([torch.atan2(v[..., 1], v[..., 0]), torch.norm(v, dim=-1)], dim=-1)


def pretty_plot_pole_figure_points(
    orientations,
    pole,
    projection="stereographic",
    crystal_symmetry="1",
    sample_symmetry="1",
    point_size=10.0,
):
    """Project and then make a pretty plot for a pole figure given points as input

    Args:
        orientations (torch.tensor or neml2.types.MRP): tensor of orientations
            as *modified* Rodrigues parameters in the *active* convention with
            arbitrary batch shape.
        pole (torch.tensor or neml2.types.Vec): pole to project, must
            broadcast with orientations

    Keyword Args:
        projection (str): which polar projection to use, options are
            "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the
            crystal symmetry to apply to the base orientations, default "1"
        sample_symmetry (str): string giving the orbifold notation for the
            sample symmetry to apply to the projected points, default "1"
        point_size (float): size of matplotlib points to plot
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import NullLocator  # noqa: PLC0415

    points = pole_figure_points(orientations, pole, projection, crystal_symmetry, sample_symmetry)
    projection = available_projections[projection]

    polar = cart2polar(points)

    # Plot
    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.scatter(polar[..., 0].cpu(), polar[..., 1].cpu(), c="k", s=point_size)

    # Make the graph nice
    plt.ylim([0, projection.limit])
    ax.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.xaxis.set_minor_locator(NullLocator())


class IPFReduction:
    """Reduce points on the sphere to a fundemental

    Keyword Args:
        v0 (torch.tensor or neml2.types.Vec): first limit
        v1 (torch.tensor or neml2.types.Vec): second limit
        v2 (torch.tensor or neml2.types.Vec): third limit
    """

    def __init__(self, v0=None, v1=None, v2=None):
        if v0 is None:
            v0 = Vec(torch.tensor([0, 0, 1.0]))
        if v1 is None:
            v1 = Vec(torch.tensor([1.0, 0, 1.0]))
        if v2 is None:
            v2 = Vec(torch.tensor([1.0, 1, 1]))
        # Do some setup
        if not isinstance(v0, Vec):
            v0 = Vec(v0)
        if not isinstance(v1, Vec):
            v1 = Vec(v1)
        if not isinstance(v2, Vec):
            v2 = Vec(v2)

        v0 = unit(v0)
        v1 = unit(v1)
        v2 = unit(v2)

        if v0.batch.ndim != 0 or v1.batch.ndim != 0 or v2.batch.ndim != 0:
            raise ValueError("v0, v1, and v2 must be single (unbatched) directions")

        self.v = [v0, v1, v2]

        n01 = cross(v0, v1)
        if vdot(n01, v2).data < 0:  # data-ok texture
            n01 = -n01

        n12 = cross(v1, v2)
        if vdot(n12, v0).data < 0:  # data-ok texture
            n12 = -n12

        n20 = cross(v2, v0)
        if vdot(n20, v1).data < 0:  # data-ok texture
            n20 = -n20

        self.n = [n01, n12, n20]

    def __call__(self, v):
        """Apply the reduction to a set of poles"""
        keep = reduce(
            torch.logical_and,
            [vdot(v, n.to(device=v.device)).data > 0 for n in self.n],  # data-ok texture
        )

        return v.batch[keep]


def inverse_pole_figure_points(
    orientations,
    direction,
    projection="stereographic",
    crystal_symmetry="1",
    sample_symmetry="1",
    reduction=None,
):
    """Project points onto an inverse pole figure

    Args:
        orientations (torch.tensor or neml2.types.MRP): tensor of orientation
            as *modified* Rodrigues parameters in the *active* convention with
            arbitrary batch shape
        direction (torch.tensor or neml2.types.Vec): pole to project, must
            broadcast with orientations

    Keyword Args:
        projection (str): which projection to use
        crystal_symmetry (str): crystal symmetry to apply
        sample_symmetry (str): sample symmetry to appy
        reduction (IPFReduction): function to reduce the poles to a fundamental
            region
    """
    if reduction is None:
        reduction = IPFReduction()
    # Do some setup
    if not isinstance(direction, Vec):
        direction = Vec(direction)
    direction = unit(direction)
    if not isinstance(orientations, MRP):
        orientations = MRP(orientations)

    # Do the projection
    sample_symmetry_operators = symmetry_operators_as_R2(
        sample_symmetry, device=orientations.device, include_inversion=True
    )
    sample_directions = rotate(direction, sample_symmetry_operators)
    # crystal directions: apply the inverse orientation (the MRP inverse is its
    # negation) as a rotation matrix
    inv_orientations = euler_rodrigues(-orientations)
    crystal_directions = rotate(sample_directions, inv_orientations.dynamic_batch.unsqueeze(-1))
    symmetry_operators = symmetry_operators_as_R2(crystal_symmetry, device=orientations.device)
    # Apply the crystal operators on a FRESH sub-batch axis so they broadcast
    # over the (already-present) sample-symmetry sub-batch axis rather than
    # colliding with it. Result carries every (orientation x crystal_op x
    # sample_op) combination, matching v2's
    # (batch,) + (crystal_sym,) + (sample_sym,) + (3,) layout.
    equivalent_directions = rotate(
        crystal_directions.sub_batch.unsqueeze(0),
        symmetry_operators.sub_batch.unsqueeze(-1),
    ).data  # data-ok texture

    # Convention keeps the upper hemisphere
    directions = Vec(equivalent_directions[equivalent_directions[..., 2] > 0])

    # Reduce to the fundamental region
    directions = reduction(directions)

    # Project
    projection = available_projections[projection]
    return projection(directions.data)  # data-ok texture


def pretty_plot_inverse_pole_figure(
    orientations,
    direction,
    projection="stereographic",
    crystal_symmetry="1",
    sample_symmetry="1",
    reduction=None,
    point_size=10.0,
    axis_labels=None,
    nline=100,
    lw=2.0,
):
    """Project and then make a pretty plot for an inverse pole figure

    Args:
        orientations (torch.tensor or neml2.types.MRP): tensor of orientation
            as *modified* Rodrigues parameters in the *active* convention with
            arbitrary batch shape
        direction (torch.tensor or neml2.types.Vec): pole to project, must
            broadcast with orientations

    Keyword Args:
        projection (str): which projection to use
        crystal_symmetry (str): crystal symmetry to apply
        sample_symmetry (str): sample symmetry to appy
        reduction (IPFReduction): function to reduce the poles to a fundamental region
        point_size (float): size of points
        axis_labels (list of str): labels for the three corners
        nline (int): resolution for drawing lines
        lw (float): line width for lines
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if reduction is None:
        reduction = IPFReduction()
    if axis_labels is None:
        axis_labels = ["100", "110", "111"]
    points = inverse_pole_figure_points(
        orientations,
        direction,
        projection,
        crystal_symmetry,
        sample_symmetry,
        reduction,
    )
    projection = available_projections[projection]

    plt.figure()
    ax = plt.subplot(111)
    ax.scatter(points.cpu()[:, 0], points.cpu()[:, 1], color="k", s=point_size)
    ax.axis("off")
    if axis_labels:
        plt.text(0.1, 0.11, axis_labels[0], transform=plt.gcf().transFigure)
        plt.text(0.86, 0.11, axis_labels[1], transform=plt.gcf().transFigure)
        plt.text(0.74, 0.88, axis_labels[2], transform=plt.gcf().transFigure)

    for i, j in ((0, 1), (1, 2), (2, 0)):
        v1 = reduction.v[i].data.cpu()  # data-ok texture
        v2 = reduction.v[j].data.cpu()  # data-ok texture
        fs = torch.linspace(0, 1, nline).cpu()
        pts = v1 * fs.unsqueeze(-1) + v2 * (1.0 - fs).unsqueeze(-1)
        pts /= torch.linalg.norm(pts, dim=-1).unsqueeze(-1)
        pts = projection(pts)
        plt.plot(pts[:, 0], pts[:, 1], "k-", lw=lw)
