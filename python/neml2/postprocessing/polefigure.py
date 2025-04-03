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

from math import sqrt

from functools import reduce

import torch

torch.set_default_dtype(torch.float64)

import matplotlib.pyplot as plt

from neml2 import crystallography
from neml2 import tensors


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


def symmetry_operators_as_R2(orbifold, device=torch.device("cpu")):
    """Return the symmetry operators for a given symmetry group as a batch of rank two tensors

    Args:
        orbifold (str): symmetry group in orbifold notation

    Keyword Args:
        device (torch.device): which device to place the tensors
    """
    return crystallography.symmetry_operations_from_orbifold(orbifold, device=device)


def pole_figure_odf(
    odf,
    pole,
    projection="stereographic",
    crystal_symmetry="1",
    nradial=20,
    ntheta=40,
    nquad=50,
    nchunk=1000,
):
    """Project an ODF onto a pole figure

    Args:
        odf (neml2.odf.ODF): ODF to project
        pole (torch.tensor or neml2.tensors.Vec): pole to project

    Keyword Args:
        projection (str): which polar projection to use, options are "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the crystal symmetry to apply to the base orientations, default "1"
        nradial (int): number of radial points to use
        ntheta (int): number of angular points to use
        nquad (int): number of points to integrate the pole figure
        nchunk (int): subbatch size to use in evaluating odf
    Returns:
        (values, mesh): tuple of the projected values and the (theta, R) mesh points used to make a contour plot
    """
    # Do some setup
    if not isinstance(pole, tensors.Vec):
        pole = tensors.Vec(pole)
    pole = pole / pole.norm()
    projection = available_projections[projection]

    # Make the grid
    r = torch.linspace(0, projection.limit, nradial, device=pole.device)
    theta = torch.linspace(0, 2 * torch.pi, ntheta, device=pole.device)
    R, T = torch.meshgrid(r, theta, indexing="ij")

    # Get points on plane
    X = R * torch.cos(T)
    Y = R * torch.sin(T)

    # Project back to sphere
    sample_poles = tensors.Vec(projection.inverse(torch.stack([X, Y], dim=-1)))

    # Get all the equivalent poles
    crystal_symmetry_operators = symmetry_operators_as_R2(crystal_symmetry, device=pole.device)
    crystal_poles = crystal_symmetry_operators * pole
    crystal_poles = crystal_poles.batch.unsqueeze(1).batch.unsqueeze(1)

    # Calculate the static rotation from crystal to sample poles
    r1 = tensors.Rot.fill_rotate_between(crystal_poles, sample_poles)

    # Calculate the rotations about the sample poles
    # Note my diabolical cleverness -- we know the arc length metric is constant in
    # the standard Rodrigues space, so we can use a special function to that gets
    # standard rodrigues parameters for the rotation about the pole and convert to MRP
    # and then drop the metric term.
    angle_values = torch.linspace(0, 2 * torch.pi, nquad + 1, device=pole.device)
    dphi = torch.diff(angle_values)
    phi = tensors.Scalar(angle_values[:-1]).batch.unsqueeze(1).batch.unsqueeze(1)
    r2 = tensors.Rot.fill_axis_angle_standard(sample_poles, phi)

    # Compose
    rm = r1.rotate(r2.batch.unsqueeze(1))

    # Calculate the ODF values
    original_shape = rm.batch.shape
    rm = rm.batch.reshape((-1,)).torch()
    A = torch.stack(
        [odf(tensors.Rot(rm[i : i + nchunk])) for i in range(0, rm.shape[0], nchunk)], dim=0
    ).reshape(original_shape)

    # Incremental rotation
    dl = (dphi).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Calculate the actual pole values
    vals = torch.mean(torch.sum(A * dl, dim=0), dim=0) / (2 * torch.pi)

    return vals, (T, R)


def pretty_plot_pole_figure_odf(
    odf,
    pole,
    projection="stereographic",
    crystal_symmetry="1",
    nradial=20,
    ntheta=40,
    nquad=50,
    nchunk=1000,
):
    """Project an ODF onto a pole figure

    Args:
        odf (neml2.odf.ODF): ODF to project
        pole (torch.tensor or neml2.tensors.Vec): pole to project

    Keyword Args:
        projection (str): which polar projection to use, options are "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the crystal symmetry to apply to the base orientations, default "1"
        nradial (int): number of radial points to use
        ntheta (int): number of angular points to use
        nquad (int): number of points to integrate the pole figure
        nchunk (int): subbatch size to use in evaluating odf
    """
    vals, mesh = pole_figure_odf(
        odf, pole, projection, crystal_symmetry, nradial, ntheta, nquad, nchunk
    )
    projection = available_projections[projection]

    # Plot
    plt.figure()
    ax = plt.subplot(111, projection="polar")
    CS = ax.contourf(mesh[0], mesh[1], vals.detach(), cmap="Greys")

    # Make the graph nice
    plt.ylim([0, projection.limit])
    ax.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.xaxis.set_minor_locator(plt.NullLocator())
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
        orientations (torch.tensor or neml2.tensors.Rot): tensor of orientations as *modified* Rodrigues
            parameters in the *active* convention with arbitrary batch shape.
        pole (torch.tensor or neml2.tensors.Vec): pole to project, must broadcast with orientations

    Keyword Args:
        projection (str): which polar projection to use, options are "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the crystal symmetry to apply to the base orientations, default "1"
        sample_symmetry (str): string giving the orbifold notation for the sample symmetry to apply to the projected points, default "1"

    Returns:
        torch.tensor: tensor with same batch shape as orientations but with end dimension (2,) giving each point
    """
    # Do some setup
    if not isinstance(pole, tensors.Vec):
        pole = tensors.Vec(pole)
    pole = pole / pole.norm()
    if not isinstance(orientations, tensors.Rot):
        orientations = tensors.Rot(orientations)

    # Get all the equivalent poles
    crystal_symmetry_operators = symmetry_operators_as_R2(
        crystal_symmetry, device=orientations.device
    )
    equivalent_poles = crystal_symmetry_operators * pole

    # Move from crystal to sample
    sample_poles = equivalent_poles.rotate(orientations.batch.unsqueeze(-1))
    # Apply sample symmetry
    sample_symmetry_operators = symmetry_operators_as_R2(
        sample_symmetry, device=orientations.device
    )
    sample_poles = sample_symmetry_operators * sample_poles.batch.unsqueeze(-1)

    # For my reference, at this point we have a tensor of (arbitrary_batch_shape,) + (crystal_symmetry,) + (sample_symmetry,) + (3,)
    sample_poles = sample_poles.torch()

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
        orientations (torch.tensor or neml2.tensors.Rot): tensor of orientations as *modified* Rodrigues
            parameters in the *active* convention with arbitrary batch shape.
        pole (torch.tensor or neml2.tensors.Vec): pole to project, must broadcast with orientations

    Keyword Args:
        projection (str): which polar projection to use, options are "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the crystal symmetry to apply to the base orientations, default "1"
        sample_symmetry (str): string giving the orbifold notation for the sample symmetry to apply to the projected points, default "1"
        point_size (float): size of matplotlib points to plot
    """
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
    ax.xaxis.set_minor_locator(plt.NullLocator())


class IPFReduction:
    """Reduce points on the sphere to a fundemental

    Keyword Args:
        v0 (torch.tensor or tensors.Vec): first limit
        v1 (torch.tensor or tensors.Vec): second limit
        v2 (torch.tensor or tensors.Vec): third limit
    """

    def __init__(
        self,
        v0=tensors.Vec(torch.tensor([0, 0, 1.0])),
        v1=tensors.Vec(torch.tensor([1.0, 0, 1.0])),
        v2=tensors.Vec(torch.tensor([1.0, 1, 1])),
    ):
        # Do some setup
        if not isinstance(v0, tensors.Vec):
            v0 = tensors.Vec(v0)
        if not isinstance(v1, tensors.Vec):
            v1 = tensors.Vec(v1)
        if not isinstance(v2, tensors.Vec):
            v2 = tensors.Vec(v2)

        v0 = v0 / v0.norm()
        v1 = v1 / v1.norm()
        v2 = v2 / v2.norm()

        self.v = [v0, v1, v2]
        self.n = [v0.cross(v1), v1.cross(v2), v2.cross(v0)]

    def __call__(self, v):
        """Apply the reduction to a set of poles"""
        keep = reduce(
            torch.logical_and,
            [v.dot(n.to(device=v.device)).torch() > 0 for n in self.n],
        )

        return tensors.Vec(v.torch()[keep])


def inverse_pole_figure_points(
    orientations,
    direction,
    projection="stereographic",
    crystal_symmetry="1",
    sample_symmetry="1",
    reduction=IPFReduction(),
):
    """Project points onto an inverse pole figure

    Args:
        orientations (torch.tensor or neml2.tensors.Rot): tensor of orientation as *modified* Rodrigues
            parameters in the *active* convention with arbitrary batch shape
        direction (torch.tensor or neml2.tensors.Vec): pole to project, must broadcast with orientations

    Keyword Args:
        projection (str): which projection to use
        crystal_symmetry (str): crystal symmetry to apply
        sample_symmetry (str): sample symmetry to appy
        reduction (IPFReduction): function to reduce the poles to a fundamental region
    """
    # Do some setup
    if not isinstance(direction, tensors.Vec):
        direction = tensors.Vec(direction)
    direction = direction / direction.norm()
    if not isinstance(orientations, tensors.Rot):
        orientations = tensors.Rot(orientations)

    # Do the projection
    sample_symmetry_operators = symmetry_operators_as_R2(
        sample_symmetry, device=orientations.device
    )
    sample_directions = sample_symmetry_operators * direction
    crystal_directions = sample_directions.rotate(orientations.inverse().batch.unsqueeze(-1))
    crystal_symmetry_operators = symmetry_operators_as_R2(
        crystal_symmetry, device=orientations.device
    )
    equivalent_directions = (
        crystal_symmetry_operators * crystal_directions.batch.unsqueeze(-1)
    ).torch()

    # Convention keeps the upper hemisphere
    directions = tensors.Vec(equivalent_directions[equivalent_directions[..., 2] > 0])

    # Reduce to the fundamental region
    directions = reduction(directions)

    # Project
    projection = available_projections[projection]
    return projection(directions.torch())


def pretty_plot_inverse_pole_figure(
    orientations,
    direction,
    projection="stereographic",
    crystal_symmetry="1",
    sample_symmetry="1",
    reduction=IPFReduction(),
    point_size=10.0,
    axis_labels=["100", "110", "111"],
    nline=100,
    lw=2.0,
):
    """Project and then make a pretty plot for an inverse pole figure

    Args:
        orientations (torch.tensor or neml2.tensors.Rot): tensor of orientation as *modified* Rodrigues
            parameters in the *active* convention with arbitrary batch shape
        direction (torch.tensor or neml2.tensors.Vec): pole to project, must broadcast with orientations

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
        v1 = reduction.v[i].torch().cpu()
        v2 = reduction.v[j].torch().cpu()
        fs = torch.linspace(0, 1, nline).cpu()
        pts = v1 * fs.unsqueeze(-1) + v2 * (1.0 - fs).unsqueeze(-1)
        pts /= torch.linalg.norm(pts, dim=-1).unsqueeze(-1)
        pts = projection(pts)
        plt.plot(pts[:, 0], pts[:, 1], "k-", lw=lw)
