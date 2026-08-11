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

r"""Elastic interaction between slip systems over one time step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....factory import register_neml2_object
from ....schema import HitSchema, dependency, derived_input, input, output, parameter
from ....types import MRP, SR2, SSR4, Scalar, euler_rodrigues, inner, rotate
from ...model import Model

if TYPE_CHECKING:
    from ....data import CrystalGeometry
    from ...chain_rule import ChainRuleDict


@register_neml2_object("SlipSystemElasticInteraction")
class SlipSystemElasticInteraction(Model):
    r"""The coupling matrix $A_{ij} = \Delta t\, M_i : \mathbb{C} : M_j$.

    Slip on system $j$ relaxes the resolved shear on system $i$ through the
    elastic response: with $\varepsilon^e = \varepsilon^{e,\rm trial} - \Delta t
    \sum_j \dot\gamma_j M_j$, the resolved shears follow $\tau_i =
    \tau^{\rm trial}_i - \sum_j A_{ij}\dot\gamma_j$. That $A$ is what turns the
    slip rule into the condensed system
    $\varphi(\dot\gamma) + A\dot\gamma = b$ that
    :class:`~neml2.models.common.CoordinateDescentPredictor` solves, with
    $b = \tau^{\rm trial}$.

    $A$ is symmetric positive semi-definite -- it is
    $\Delta t\,B^{\mathsf T}\mathbb{C}B$ with $B$ the Schmid map and
    $\mathbb{C}$ the elastic stiffness -- and singular, because the Schmid map
    sends $\mathbb{R}^{n_{\rm slip}}$ into a lower-dimensional space of
    symmetric tensors (the classical Taylor ambiguity). Both properties are
    relied on downstream: positive semi-definiteness is what brackets each
    coordinate solve, and singularity is harmless because the slip rule's own
    monotone nonlinearity is what makes the condensed system strictly convex.

    It is emitted as a :class:`~neml2.types.Scalar` carried on two sub-batch
    axes, ``(n_slip, n_slip)`` -- a matrix of scalars is exactly what the
    sub-batch machinery represents.

    **Frames.** Everything is contracted in the lab frame, matching the rest of
    the crystal-plasticity library: ``ResolvedShear`` rotates the Schmid tensors
    and contracts them with the lab-frame stress, and ``GeneralElasticity``
    rotates the stiffness (``T_rot = rotate(T, R)``) before applying it to the
    lab-frame strain. So both Schmid operands *and* the stiffness are rotated
    here. Getting this wrong yields a plausible but silently wrong matrix, so it
    is pinned by a test: for an isotropic stiffness, $A$ must not depend on the
    orientation at all.
    """

    hit = HitSchema(
        input("orientation", MRP, "Active-convention orientation", attr="_q"),
        input("time", Scalar, "Time", default="t", attr="_t"),
        derived_input("time", Scalar, attr="_tn", suffix="~1"),
        output(
            "coupling",
            Scalar,
            "The coupling matrix, sub-batched over (n_slip, n_slip)",
            attr="_A",
        ),
        parameter(
            "elastic_stiffness_tensor",
            SSR4,
            "Elastic stiffness tensor, in the crystal frame",
            attr="T",
            allow_promotion=True,
        ),
        dependency(
            "crystal_geometry",
            "get_data",
            "The data object carrying the crystallographic information",
            default="crystal_geometry",
        ),
    )

    _q: str
    _t: str
    _tn: str
    _A: str
    T: SSR4

    def __init__(
        self,
        *,
        crystal_geometry: CrystalGeometry,
        elastic_stiffness_tensor,
    ) -> None:
        # A custom __init__ (needed for the crystal_geometry dependency) opts out
        # of the schema's automatic parameter declaration, so declare it here --
        # the same call `from_hit` would have made.
        super().__init__()
        self._cg = crystal_geometry
        self.declare_typed_parameter("T", elastic_stiffness_tensor, SSR4, allow_promotion=True)

    def forward(  # type: ignore[override]
        self,
        q: MRP,
        t: Scalar,
        tn: Scalar,
        *promoted_params,
        v: ChainRuleDict | None = None,
    ):
        del v  # consumed only by a predictor, which is never differentiated
        T = self._get_param("T", promoted_params, SSR4)
        R = euler_rodrigues(q)
        # Rotate both the stiffness and the Schmid tensors into the lab frame --
        # the convention GeneralElasticity and ResolvedShear respectively use.
        T_rot = rotate(T, R)
        M: SR2 = rotate(self._cg.M, R.sub_batch.unsqueeze(-1))
        CM: SR2 = T_rot.sub_batch.unsqueeze(-1) @ M

        # Outer product over slip systems: put i on one new sub-batch axis and j
        # on another, then contract the base. Result is sub-batched (i, j).
        Mi = M.sub_batch.unsqueeze(-1)
        CMj = CM.sub_batch.unsqueeze(-2)
        return (t - tn) * inner(Mi, CMj)


__all__ = ["SlipSystemElasticInteraction"]
