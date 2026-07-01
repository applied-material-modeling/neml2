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

"""Python-native port of the v2 C++ ``MazarsEquivalentStrain`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output
from ....types import SR2, Scalar, inner, macaulay, norm
from ....types.functions import r2_from_sr2
from ....types.linalg import diag, eigh, transpose
from ...chain_rule import ChainRuleDict
from ...model import Model

# Numerical guard for 1/ε̃ when ε̃ → 0 (no positive principal strains,
# e.g., uniaxial-strain compression with no Poisson laterals). Adding this
# floor changes ε̃ at most by 1e-30 in absolute terms and yields the
# mathematically-correct gradient (zero) when there are no positive
# principal strains.
_EPS_FLOOR = 1.0e-30


@register_neml2_object("MazarsEquivalentStrain")
class MazarsEquivalentStrain(Model):
    r"""Mazars equivalent strain -- Mazars (1986), Section 3.2:

    .. math::
        \tilde{\varepsilon} = \sqrt{\sum_{i=1}^{3} \langle \varepsilon_i \rangle^2}

    where :math:`\varepsilon_i` are the principal strains and
    :math:`\langle x \rangle = \max(x, 0)` is the Macaulay bracket.

    Only positive (tensile) principal strains contribute, so pure
    compressive loading without Poisson-induced laterals gives
    :math:`\tilde{\varepsilon} = 0`. This quantity drives damage onset
    in Mazars-family CDM models.
    """

    hit = HitSchema(
        input("strain", SR2, "Total strain (SR2)"),
        output("equivalent_strain", Scalar, "Mazars equivalent strain (Scalar)"),
    )

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # ---- Primal ----
        # 1. Principal strains (Vec) and principal directions (R2).
        # 2. Element-wise Macaulay bracket — keep positive principals only.
        # 3. Euclidean norm over the 3-vector → Scalar equivalent strain.
        eigvals, eigvecs = eigh(strain)
        pos = macaulay(eigvals)
        eq_strain = norm(pos)

        if v is None:
            return eq_strain

        # ---- Chain rule ----
        # Closed-form JVP. Derivation:
        #   ε̃²  = Σ ⟨λᵢ⟩²
        #   ε̃   = sqrt(ε̃²)
        #   ∂ε̃/∂λᵢ        = ⟨λᵢ⟩ / ε̃              (the H(λᵢ) factor is implicit:
        #                                            ⟨λᵢ⟩ = 0 when λᵢ ≤ 0)
        #   ∂λᵢ/∂S        = vᵢ ⊗ vᵢ               (eigenvalue derivative — well-defined
        #                                            even at degenerate eigenvalues)
        # Chain together:
        #   ∂ε̃/∂S : V_S  = (1/ε̃) · Σ ⟨λᵢ⟩ · (vᵢᵀ V_S vᵢ)
        #
        # The inner sum is computable cleanly using typed primitives:
        #   diag(Vᵀ · V_S · V) = (v_iᵀ V_S v_i)_i           (a Vec of 3)
        #   inner(⟨λ⟩₊, that)  = Σ ⟨λᵢ⟩ · (v_iᵀ V_S v_i)    (a Scalar)
        # then divide by ε̃ — with a tiny epsilon floor so ε̃ → 0 returns 0
        # rather than NaN.
        eq_safe = eq_strain + _EPS_FLOOR

        def strain_action(
            V_S: SR2,
            eigvecs: SR2 = eigvecs,  # noqa: type narrowed at runtime
            pos: SR2 = pos,
            eq_safe: SR2 = eq_safe,
        ) -> Scalar:
            V_S_full = r2_from_sr2(V_S)
            rotated = transpose(eigvecs) @ V_S_full @ eigvecs
            return inner(pos, diag(rotated)) / eq_safe

        return eq_strain, self.apply_chain_rule(
            v, "equivalent_strain", {"strain": strain_action}, output=eq_strain
        )
