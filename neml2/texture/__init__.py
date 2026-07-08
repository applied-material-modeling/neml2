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

"""Crystallographic-texture postprocessing: ODFs and (inverse) pole figures.

Ported from v2 ``neml2.postprocessing``. Orientations are
:class:`~neml2.types.MRP`, directions :class:`~neml2.types.Vec`, and symmetry
operators :class:`~neml2.types.R2`; the typed orientation-space ops from
:mod:`neml2.types` and :func:`neml2.ops.symmetry` carry the data flow.
"""

from neml2.texture.odf import (
    KDEODF,
    ODF,
    DeLaValleePoussinKernel,
    Kernel,
    beta,
    gauss_points,
    rotation_quadrature,
    spherical_quadrature,
    split,
)
from neml2.texture.polefigure import (
    IPFReduction,
    LambertProjection,
    StereographicProjection,
    arbitrary_hemispherical_quadrature,
    available_projections,
    cart2polar,
    inverse_pole_figure_points,
    pole_figure_odf,
    pole_figure_points,
    pretty_plot_inverse_pole_figure,
    pretty_plot_pole_figure_odf,
    pretty_plot_pole_figure_points,
    symmetry_operators_as_R2,
)

__all__ = [
    # odf
    "ODF",
    "KDEODF",
    "Kernel",
    "DeLaValleePoussinKernel",
    "beta",
    "gauss_points",
    "spherical_quadrature",
    "rotation_quadrature",
    "split",
    # polefigure
    "StereographicProjection",
    "LambertProjection",
    "available_projections",
    "symmetry_operators_as_R2",
    "arbitrary_hemispherical_quadrature",
    "pole_figure_odf",
    "pole_figure_points",
    "inverse_pole_figure_points",
    "IPFReduction",
    "pretty_plot_pole_figure_odf",
    "pretty_plot_pole_figure_points",
    "pretty_plot_inverse_pole_figure",
    "cart2polar",
]
