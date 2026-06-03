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

"""Shared type aliases for chain-rule sensitivity propagation.

Two orders are carried separately:

* ``ChainRuleDict`` (first order) — ``v[var_name][seed_leaf]`` is a typed
  wrapper for ``var_name`` carrying the seed-direction axis K as the leftmost
  batch dim: ``(K, *B, *sub, *BASE_SHAPE)``.

* ``SecondOrderChainRuleDict`` — ``v2[var_name][seed_a][seed_b]`` is a typed
  wrapper for ``var_name`` carrying two leading seed axes:
  ``(N_a, N_b, *B, *sub, *BASE_SHAPE)``. Both orderings are stored explicitly
  (no symmetry shortcut); the Inductor pipeline would have to recover symmetry
  anyway and the redundant entries also let upstream asymmetric chain-rule
  expressions stay simple.

The v2 path is opt-in: only models that may appear inside a
:class:`~neml2.models.solid_mechanics.plasticity.Normality` wrap implement v2 actions,
and only those models accept a ``v2`` kwarg in their ``forward``. Sub-batch
leaves never opt in, so the existing ``ComposedModel`` guard
(``SUPPORTS_SECOND_ORDER=False``) already prevents the combination.

Sub-batch sparsity flags
------------------------

``Model.list_deriv: ListDerivSpec`` declares the per-``(output, input)``
sub-batch dependence pattern for the leaf's derivative:

* ``"diagonal"`` (default for any pair absent from the dict) — site ``i`` of
  the output depends only on site ``i`` of the input. The leading-K typed
  tangent contract carries this naturally because sub-batch axes remain in the
  wrapper's normal sub-batch region and every K slice acts independently per
  site.

* ``"dense"`` — site ``i`` of the output depends on multiple sites of the
  input (reductions, FV cell coupling, linear-stencil ops). The leaf is
  responsible for writing the right contraction in its chain-rule action;
  the only thing the flag affects is the composed-model resolver, which
  combines flags transitively and uses the result to decide whether the
  end-to-end equation system can be solved with a BLOCK-structured (Schur)
  solver or must fall back to DENSE.

The flag does NOT change the shape contract of
:meth:`Model.apply_chain_rule` — the leaf's action signature is identical
in both cases.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

from torch import Tensor

from neml2.types._base import TensorWrapper

# tangents flow as ordinary typed wrappers (``Scalar`` / ``SR2``
# / ``R2`` / …) with the K (seed-direction) axis carried as the **leftmost** batch
# dimension. A tangent of type ``T`` *is* a ``T``, so leaf action functions receive
# a typed wrapper and return one, and every operation is the same primal wrapper
# algebra (``vol`` / ``dev`` / ``inner`` / ``+`` / ``sub_batch_sum`` / …) — K rides
# through as a broadcast batch dim. D-062 forbids leaf-level full-Jacobian
# matmul actions;
# irreducible local derivative contractions live behind named typed JVP primitives.
TangentAction: TypeAlias = TensorWrapper
ChainRuleDict: TypeAlias = dict[str, dict[str, TensorWrapper]]
# The incoming tangent is the input variable's concrete type (a ``Scalar``-tangent
# *is* a ``Scalar``, etc.), so leaf actions annotate their parameter with that
# concrete type; ``...`` keeps the alias assignable from any such signature.
ChainRuleAction: TypeAlias = Callable[..., TensorWrapper]

# second-order tangents in transit through the framework
# carry TWO leading axes (N_a, N_b, *dyn, *sub, *base) — the first-order
# leading-K axis split in two for the two outer seed directions.
SecondOrderTangentAction: TypeAlias = TensorWrapper
SecondOrderChainRuleDict: TypeAlias = dict[str, dict[str, dict[str, TensorWrapper]]]
# the leaf-side ``action_2(Va, Vb)`` receives ONE
# tangent slice per slot — both shaped like the primal
# ``(*dyn, *sub, *base_i)``, ``(*dyn, *sub, *base_j)``, NO leading seed dim —
# and returns a primal-shape ``(*dyn, *sub, *base_out)`` bilinear. The
# framework (``Model.apply_chain_rule_2`` via ``_apply_action_2``) iterates
# the ``(N_a, N_b)`` seed-pair outer and stacks results. This keeps every
# leaf's Hessian body in pure typed-wrapper algebra (e.g.
# ``2 * Va * Vb``, ``inner(Va, dev(Vb))``, ``coef * Va * Vb``) — no
# ``.data`` access, no explicit outer-product unsqueezing.
SecondOrderChainRuleAction: TypeAlias = Callable[..., TensorWrapper]

#: Per-(output, input) sub-batch sparsity flag — see module docstring.
SparsityFlag: TypeAlias = Literal["diagonal", "dense"]

#: Sparse declaration: only pairs that DIFFER from the default ``"diagonal"``
#: need to appear. Absent pairs are treated as ``"diagonal"`` (structural
#: zeros — pairs with no dependency at all — never appear in this dict).
ListDerivSpec: TypeAlias = dict[tuple[str, str], SparsityFlag]


def combine_sparsity(*flags: SparsityFlag) -> SparsityFlag:
    """Combine sparsity flags along a chain or across parallel paths.

    ``"dense"`` is absorbing: a chain stays sparse only if every step is
    diagonal, and parallel paths converge to dense if any path is dense.
    """
    return "dense" if any(f == "dense" for f in flags) else "diagonal"


def matvec(M: Tensor, v: Tensor) -> Tensor:
    """Fused matrix-vector product without ``einsum``.

    ``M @ v`` where the trailing dims are ``(..., n_row, n_col)`` and
    ``(..., n_col)`` respectively. Implemented as
    ``(M @ v.unsqueeze(-1)).squeeze(-1)`` so Inductor can fuse the trailing
    pointwise ops around the matmul. Equivalent to
    ``torch.einsum("...ij,...j->...i", M, v)`` but exposes the matmul to the
    scheduler directly.
    """
    return (M @ v.unsqueeze(-1)).squeeze(-1)


__all__ = [
    "TangentAction",
    "ChainRuleDict",
    "ChainRuleAction",
    "SecondOrderTangentAction",
    "SecondOrderChainRuleDict",
    "SecondOrderChainRuleAction",
    "SparsityFlag",
    "ListDerivSpec",
    "combine_sparsity",
    "matvec",
]
