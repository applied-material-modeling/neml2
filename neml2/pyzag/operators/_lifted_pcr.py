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

"""Lifted-arrowhead parallel cyclic reduction (PCR) for the neml2 pyzag backend.

For a per-timestep block with the *arrowhead* structure of a per-site (BLOCK)
group coupled to a shared global (DENSE) group, the PCR reduced subdiagonal
``B' = -B A^{-1} B`` is dense across sites -- it cannot live in the per-site
``(N, np, np)`` storage.  This module carries that dense operator implicitly as
the Schur complement of a slightly larger *sparse* arrowhead ``T = {D, U, W, V}``
(``Schur(T) = D + U W V``), folds it with a ``compose`` product, and applies it
by a gather--core--scatter, so nothing of size ``N x N`` in sites is ever built.
The dense group grows only with the fold depth (``q = hops * ns``), independent
of the site count ``N``.

The public entry point is :class:`NEML2LiftedPCRFactorization`, a drop-in
``pyzag.chunktime.BidiagonalInverseOperator`` selected by
:mod:`neml2.pyzag.operators._jacobian` when PCR is requested on a layout with an
intermediate (BLOCK) dimension.
"""

from __future__ import annotations

from typing import cast

import torch
from pyzag.chunktime import BidiagonalInverseOperator
from pyzag.operators.base import BlockVector

from ._operator import NEML2SolvableBlockOperator
from ._vector import NEML2BlockVector

# ---------------------------------------------------------------------------
# Lift algebra.  A lift is a dict with tensors carrying leading (batch,) dims:
#   D : (*b, N, np, np)   site-diagonal
#   U : (*b, N, np, q)    left arm  (site -> global bottleneck)
#   W : (*b, q, q)        shared core (no site axis)
#   V : (*b, N, q, np)    right arm (global bottleneck -> site)
# and represents the primary (site) operator  Schur = D + U W V.  q == 0 allowed.
# Site axis is dim -3 for D/U/V; W has no site axis.
# ---------------------------------------------------------------------------


def _empty_arm_U(D):
    """Zero-width left arm ``U`` (*b, N, np, 0) matching ``D``'s batch/site/dtype."""
    return D.new_zeros(D.shape[:-1] + (0,))  # (*b, N, np, 0)


def _empty_arm_V(D):
    """Zero-width right arm ``V`` (*b, N, 0, np) matching ``D``."""
    return D.new_zeros(D.shape[:-2] + (0, D.shape[-1]))  # (*b, N, 0, np)


def _empty_core(D):
    """Zero-size core ``W`` (*b, 0, 0) matching ``D`` (a q=0 lift)."""
    return D.new_zeros(D.shape[:-3] + (0, 0))  # (*b, 0, 0)


def lift_site_diagonal(D):
    """Lift a pure site-diagonal operator ``D`` (*b, N, np, np) with q=0."""
    return dict(D=D, U=_empty_arm_U(D), W=_empty_core(D), V=_empty_arm_V(D))


def matvec_p(T, up):
    """Apply Schur(T) to a per-site vector ``up`` (*b, N, np) -> (*b, N, np).

    Gather over sites through V, hit the q-dim core W, scatter back through U,
    plus the site-diagonal D.  Never forms an N x N object.
    """
    D, U, W, V = T["D"], T["U"], T["W"], T["V"]
    out = torch.matmul(D, up.unsqueeze(-1)).squeeze(-1)  # D @ up per site
    if W.shape[-1] > 0:
        Vu = torch.matmul(V, up.unsqueeze(-1)).squeeze(-1).sum(dim=-2)  # gather -> (*b, q)
        WVu = torch.matmul(W, Vu.unsqueeze(-1)).squeeze(-1)  # core -> (*b, q)
        # scatter: U^i @ WVu, broadcast WVu over the site axis
        scatter = torch.matmul(U, WVu.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
        out = out + scatter
    return out


def compose(P, Q):
    """Factored product: returns a lift of Schur(P) @ Schur(Q)  (Thm 1)."""
    DP, UP, WP, VP = P["D"], P["U"], P["W"], P["V"]
    DQ, UQ, WQ, VQ = Q["D"], Q["U"], Q["W"], Q["V"]
    qP, qQ = WP.shape[-1], WQ.shape[-1]

    D = torch.matmul(DP, DQ)  # DP DQ (site-diagonal)
    U = torch.cat([UP, torch.matmul(DP, UQ)], dim=-1)  # [UP | DP UQ]
    V = torch.cat([torch.matmul(VP, DQ), VQ], dim=-2)  # [VP DQ ; VQ]

    # site-summed cross contraction VP.UQ = sum_i VP^i UQ^i -> (*b, qP, qQ)
    VPUQ = torch.matmul(VP, UQ).sum(dim=-3)
    cross = torch.matmul(torch.matmul(WP, VPUQ), WQ)  # (*b, qP, qQ)
    top = torch.cat([WP, cross], dim=-1)  # (*b, qP, qP+qQ)
    zero_bl = WP.new_zeros(WP.shape[:-2] + (qQ, qP))
    bot = torch.cat([zero_bl, WQ], dim=-1)  # (*b, qQ, qP+qQ)
    W = torch.cat([top, bot], dim=-2)
    return dict(D=D, U=U, W=W, V=V)


def neg(T):
    """-Schur(T) = -(D + U W V) = (-D) + U (-W) V."""
    return dict(D=-T["D"], U=T["U"], W=-T["W"], V=T["V"])


# ---------------------------------------------------------------------------
# Arrowhead-cell extraction from a two-group (BLOCK + DENSE) AssembledMatrix.
# ---------------------------------------------------------------------------


def _group_roles(layout):
    """Return (block_group_index, dense_group_index_or_None)."""
    blk = dense = None
    for g in range(layout.ngroup):
        if layout.structure[g] == "block":
            if blk is None:
                blk = g
        elif dense is None:
            dense = g
    if blk is None:
        raise NotImplementedError(
            "Lifted PCR requires a BLOCK (per-site) group; got layout with none."
        )
    return blk, dense


def _diagonal_cells(am):
    """Extract ``(App, Aps, Asp, Ass, ns)`` from a diagonal AM.

    When there is no DENSE group (``ns == 0``) the arm/core tensors are ``None``
    and the reduction degenerates to per-site-independent PCR.
    """
    p, s = _group_roles(am.row_layout)
    App = am.tensors[p][p].data  # data-ok pyzag boundary
    if s is None:
        return App, None, None, None, 0
    Aps = am.tensors[p][s].data  # data-ok pyzag boundary
    Asp = am.tensors[s][p].data  # data-ok pyzag boundary
    Ass = am.tensors[s][s].data  # data-ok pyzag boundary
    return App, Aps, Asp, Ass, Ass.shape[-1]


def _subdiag_Bpp(am):
    """Extract the site-diagonal Bpp (*, N, np, np) of a subdiagonal AM.

    The subdiagonal couples old-state (BLOCK/site) to the residual.  For the
    supported (Taylor-like) case the DENSE residual rows and any DENSE old-state
    columns are zero, so the subdiagonal is a pure site->site operator Bpp.
    """
    prow = am.row_layout
    pcol = am.col_layout
    p_row, s_row = _group_roles(prow)
    p_col, _ = _group_roles(pcol)
    Bpp = am.tensors[p_row][p_col].data  # data-ok pyzag boundary
    # Guard: the DENSE-residual rows of the subdiagonal must vanish (no global
    # time history), otherwise this simplified path would be silently wrong.
    if s_row is not None:
        Bsp = am.tensors[s_row][p_col].data  # data-ok pyzag boundary
        if Bsp.abs().max() > 1e-10 * (Bpp.abs().max() + 1e-30):
            raise NotImplementedError(
                "Lifted PCR currently supports a site-only subdiagonal (Bsp == 0). "
                "The global residual appears to depend on old state; the general "
                "q_B > 0 case is not yet implemented."
            )
    return Bpp


# ---------------------------------------------------------------------------
# Per-block diagonal factorization (Step 1 / Lemma 1).
# ---------------------------------------------------------------------------


class _DiagFactors:
    """Factored A^{-1} for one diagonal block: App^{-1}, Y, Z, S, and the lift Ahat."""

    __slots__ = ("App", "App_inv", "Aps", "Asp", "Ass", "Y", "Z", "S", "S_inv", "ns", "Ahat")

    def __init__(self, App, Aps, Asp, Ass, ns):
        """Factor one arrowhead diagonal block into its Schur pieces and the lift ``Ahat``.

        With ``ns == 0`` (no DENSE group) only ``App^{-1}`` is formed and ``Ahat``
        is the plain site-diagonal lift; otherwise the Schur complement ``S`` and
        the arms ``Y``/``Z`` give ``(A^{-1})_pp = App^{-1} + Y S^{-1} Z`` as a lift.
        """
        self.App = App
        self.App_inv = torch.linalg.inv(App)  # per-site inverse (batched)
        self.ns = ns
        if ns == 0:
            self.Aps = self.Asp = self.Ass = None
            self.Y = self.Z = self.S = self.S_inv = None
            self.Ahat = lift_site_diagonal(self.App_inv)
            return
        assert Aps is not None and Asp is not None and Ass is not None
        self.Aps, self.Asp, self.Ass = Aps, Asp, Ass
        self.Y = torch.matmul(self.App_inv, Aps)  # (*b, N, np, ns)
        self.Z = torch.matmul(Asp, self.App_inv)  # (*b, N, ns, np)
        self.S = Ass - torch.matmul(Asp, self.Y).sum(dim=-3)  # (*b, ns, ns)
        self.S_inv = torch.linalg.inv(self.S)
        # (A^-1)_pp = App^-1 + Y S^-1 Z  ->  lift
        self.Ahat = dict(D=self.App_inv, U=self.Y, W=self.S_inv, V=self.Z)


def _solve_arrowhead(fac: _DiagFactors, vp, vs):
    """Full arrowhead solve A^{-1} [vp; vs] -> (xp, xs) via the 6-step Schur.

    vp : (*b, N, np),  vs : (*b, ns) or None (ns == 0).
    """
    t = torch.matmul(fac.App_inv, vp.unsqueeze(-1)).squeeze(-1)  # App^-1 vp
    if fac.ns == 0:
        return t, None
    assert fac.Asp is not None and fac.S_inv is not None and fac.Y is not None and vs is not None
    rhs_s = vs - torch.matmul(fac.Asp, t.unsqueeze(-1)).squeeze(-1).sum(dim=-2)
    xs = torch.matmul(fac.S_inv, rhs_s.unsqueeze(-1)).squeeze(-1)
    xp = t - torch.matmul(fac.Y, xs.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
    return xp, xs


# ---------------------------------------------------------------------------
# The factorization.
# ---------------------------------------------------------------------------


class NEML2LiftedPCRFactorization(BidiagonalInverseOperator):
    """Exact O(N) parallel cyclic reduction for arrowhead (BLOCK[+DENSE]) blocks.

    Carries the reduced subdiagonal as a lifted arrowhead (Schur complement) and
    folds it with :func:`compose`, so the dense per-site operator is never
    manifested.  ``A`` is the diagonal operator (arrowhead), ``B`` the
    subdiagonal; both are :class:`NEML2SolvableBlockOperator`.
    """

    def __init__(self, A, B, *args, **kwargs) -> None:
        """Store the diagonal ``A`` and subdiagonal ``B`` operators (both must be neml2-backed)."""
        super().__init__(A, B, *args, **kwargs)
        if not isinstance(A, NEML2SolvableBlockOperator):
            raise TypeError("NEML2LiftedPCRFactorization requires a NEML2SolvableBlockOperator A.")
        if not isinstance(B, NEML2SolvableBlockOperator):
            raise TypeError("NEML2LiftedPCRFactorization requires a NEML2SolvableBlockOperator B.")

    def matvec(self, v: BlockVector) -> NEML2BlockVector:
        """Apply the inverse bidiagonal chunk operator to ``v`` via lifted PCR.

        Extracts the arrowhead diagonal cells and the site-only subdiagonal
        ``Bpp`` once, seeds each block's lift ``T[k] = lift(Bpp[k])``, then runs
        stride-doubling cyclic reduction: at each level the right-hand side is
        updated by :func:`matvec_p` and the subdiagonal is folded via
        ``neg(compose(compose(T[k], Ahat[k-h]), T[k-h]))``. A final per-block
        :func:`_solve_arrowhead` recovers ``(xp, xs)``, which are reassembled into
        a :class:`~neml2.pyzag.operators.NEML2BlockVector`. Handles both the
        two-group (BLOCK+DENSE) and degenerate single-group BLOCK (``ns == 0``) cases.
        """
        if not isinstance(v, NEML2BlockVector):
            raise TypeError("NEML2LiftedPCRFactorization.matvec expects a NEML2BlockVector.")

        # self.A / self.B are typed as the base BlockOperator by pyzag; __init__
        # validated they are NEML2SolvableBlockOperator, so narrow for .am access.
        A = cast(NEML2SolvableBlockOperator, self.A)
        B = cast(NEML2SolvableBlockOperator, self.B)
        n = A.nblk
        Bpad = B.pad_front(1)  # subdiagonal, nblk blocks; B[0] is a zero pad

        # ----- extract cells once (time axis 0), then index per block -----
        App_f, Aps_f, Asp_f, Ass_f, ns = _diagonal_cells(A.am)
        Bpp_full = _subdiag_Bpp(Bpad.am)  # (n,*b,N,np,np)

        def _fac(k):
            if ns == 0:
                return _DiagFactors(App_f[k], None, None, None, 0)
            assert Aps_f is not None and Asp_f is not None and Ass_f is not None
            return _DiagFactors(App_f[k], Aps_f[k], Asp_f[k], Ass_f[k], ns)

        fac = [_fac(k) for k in range(n)]
        T: list[dict[str, torch.Tensor]] = [lift_site_diagonal(Bpp_full[k]) for k in range(n)]

        # ----- disassemble the rhs into per-block (vp, vs) -----
        p_grp, s_grp = _group_roles(v.layout)
        vp_full = v.raw_tensors[p_grp]  # (nblk, batch, N, np)
        vs_full = v.raw_tensors[s_grp] if s_grp is not None else None
        vp = [vp_full[k] for k in range(n)]
        vs: list[torch.Tensor | None] = [
            vs_full[k] if vs_full is not None else None for k in range(n)
        ]

        # ----- cyclic reduction (stride doubling) -----
        h = 1
        while h < n:
            newT: list[dict[str, torch.Tensor]] = []
            newvp = list(vp)
            newvs = list(vs)
            for k in range(n):
                if k - h >= 0:
                    up, _us = _solve_arrowhead(fac[k - h], vp[k - h], vs[k - h])
                    newvp[k] = vp[k] - matvec_p(T[k], up)  # B pp-only: only vp changes
                    folded = neg(compose(compose(T[k], fac[k - h].Ahat), T[k - h]))
                    newT.append(
                        folded
                        if (k - 2 * h >= 0)
                        else lift_site_diagonal(torch.zeros_like(T[k]["D"]))
                    )
                else:
                    newT.append(lift_site_diagonal(torch.zeros_like(T[k]["D"])))
            T, vp, vs = newT, newvp, newvs
            h *= 2

        # ----- final per-block diagonal solve and reassembly -----
        xp_blocks = []
        xs_blocks = []
        for k in range(n):
            xp, xs = _solve_arrowhead(fac[k], vp[k], vs[k])
            xp_blocks.append(xp)
            if xs is not None:
                xs_blocks.append(xs)

        out_tensors = list(v.raw_tensors)
        out_tensors[p_grp] = torch.stack(xp_blocks, dim=0)
        if s_grp is not None:
            out_tensors[s_grp] = torch.stack(xs_blocks, dim=0)
        return NEML2BlockVector(out_tensors, v.layout, list(v.intmd_dims))


__all__ = ["NEML2LiftedPCRFactorization", "compose", "matvec_p", "neg", "lift_site_diagonal"]
