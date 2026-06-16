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

"""V2P-1 pre-tests: end-to-end shape-based detection through the new chain rule.

These tests exercise the canonical no-classification seed
(``broadcast K_paired`` for sub_batch axes, ``full K_base`` for base
axes), elementwise compactness preservation, mean-exposing, cross-mix
auto-fullify, and FD-Jacobian comparison against
``torch.autograd.functional.jacobian``.

RED on clean HEAD — they turn green incrementally as the K-state
machinery (V2P-2), reduction primitives (V2P-4), seed gen (V2P-5),
and leaf audit (V2P-8) land. Test imports happen inside test bodies so
a missing symbol surfaces as a single-test failure.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Canonical no-classification seed (plan §"v3 adaptation")
# ---------------------------------------------------------------------------


def test_seed_shape_for_scalar_with_grain_axis() -> None:
    """For a Scalar input with sub_batch_shape=(N_grain,), the canonical
    seed is broadcast K_paired (size 1, paired with sub axis 0); no
    K_base (Scalar has base_size=())."""
    from neml2.es._helpers import _expanded_identity_seed
    from neml2.types import Scalar

    # Primal: Scalar with dyn=(3,) and sub_batch=(N_grain=5,)
    x = Scalar(torch.zeros(3, 5), sub_batch_ndim=1)
    seed = _expanded_identity_seed(x)
    # K_paired only: (1, 3, 5)
    assert seed.k_ndim == 1
    assert seed.k_state == ("broadcast",)
    assert seed.k_pairing == (0,)
    assert seed.data.shape == (1, 3, 5)


def test_seed_shape_for_sr2_with_grain_axis() -> None:
    """For an SR2 input with sub_batch_shape=(N_grain,) and base_size=(6,),
    the canonical seed has K_paired (size 1, paired) AND K_base (size 6,
    full, unpaired)."""
    from neml2.es._helpers import _expanded_identity_seed
    from neml2.types import SR2

    # Primal: SR2 with dyn=(3,) and sub_batch=(N_grain=5,)
    x = SR2(torch.zeros(3, 5, 6), sub_batch_ndim=1)
    seed = _expanded_identity_seed(x)
    # K_paired + K_base: (1, 6, 3, 5, 6)
    assert seed.k_ndim == 2
    assert seed.k_state == ("broadcast", "full")
    assert seed.k_pairing == (0, None)
    assert seed.data.shape == (1, 6, 3, 5, 6)


# ---------------------------------------------------------------------------
# Compactness preservation through elementwise ops
# ---------------------------------------------------------------------------


def test_elementwise_scalar_multiply_preserves_paired_broadcast() -> None:
    """Elementwise ops on a broadcast-K-paired tangent preserve the
    paired-broadcast K state — no inflation."""
    from neml2.types import Scalar

    seed = Scalar(
        torch.ones(1, 3, 1),  # K=1, dyn=3, sub=1 (paired-broadcast)
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(5,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    out = seed * 2.0
    # K still compact-paired; no inflation.
    assert out.k_ndim == 1
    assert out.k_state == ("broadcast",)
    assert out.k_pairing == (0,)
    assert out.data.shape == (1, 3, 1)
    assert torch.allclose(out.data, torch.full((1, 3, 1), 2.0))


def test_elementwise_binary_combines_k_state() -> None:
    """``a + b`` propagates the combined K state per the lattice."""
    from neml2.types import Scalar

    # a: paired-broadcast on sub axis 0
    a = Scalar(
        torch.ones(1, 3, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(5,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    # b: same shape — both compact, both paired.
    b = Scalar(
        torch.ones(1, 3, 1) * 3.0,
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(5,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    out = a + b
    # Both broadcast-paired on axis 0 → result broadcast-paired on axis 0.
    assert out.k_ndim == 1
    assert out.k_state == ("broadcast",)
    assert out.k_pairing == (0,)
    assert out.data.shape == (1, 3, 1)


# ---------------------------------------------------------------------------
# Mean-reducing exposes K
# ---------------------------------------------------------------------------


def test_mean_over_paired_sub_axis_exposes_k_to_full_size() -> None:
    """Mean-reducing a sub_batch axis paired with a broadcast K axis
    promotes K from size-1 broadcast to size-N full, drops the K pairing,
    drops the sub axis, AND divides by N."""
    from neml2.types import Scalar
    from neml2.types.functions import mean

    seed = Scalar(
        torch.ones(1, 5),  # K=1, sub=5 (no dyn)
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(5,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    out = mean(seed.sub_batch, dim=0)
    assert out.k_ndim == 1
    assert out.k_state == ("full",)
    assert out.k_pairing == (None,)
    assert out.sub_batch_ndim == 0
    assert out.data.shape == (5,)
    # /N scaling — each K row carries 1/N.
    assert torch.allclose(out.data, torch.full((5,), 1.0 / 5.0))


# ---------------------------------------------------------------------------
# Cross-mixing leaf must call fullify before contracting paired sub axis
# ---------------------------------------------------------------------------


def test_inner_product_over_paired_sub_axis_requires_fullify() -> None:
    """An inner product that contracts the paired sub_batch axis must
    operate on a fullified tangent — otherwise it sums over a size-1
    placeholder and silently produces the wrong answer."""
    from neml2.types import Tensor as DynTensor
    from neml2.types.functions import fullify, sum_sub_batch

    # Tangent: K_paired broadcast on sub axis 0 (size 4), no other axes.
    t = DynTensor(
        torch.ones(1, 4),  # data shape (K=1, sub=4)
        batch_ndim=0,
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(4,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    # ``fullify`` is generic over ``_TW: TensorWrapper``; the dynamic-base
    # ``Tensor`` is intentionally a separate class hierarchy (variable
    # base_ndim) but is duck-typed compatibly for chain-rule helpers. The
    # runtime works; the type system can't unify the two trees.
    full = fullify(t)  # pyright: ignore[reportArgumentType]
    # Post-fullify: K=4 full, sub_pair dropped to identity on (K, sub).
    assert full.k_state == ("full",)
    assert full.k_pairing == (None,)
    # Sum over the sub axis on the fullified tangent — total contraction.
    out = sum_sub_batch(full, axis=0)
    # Each K row was the eye row — sum is 1 per row.
    assert out.data.shape == (4,)
    assert torch.allclose(out.data, torch.ones(4))


# ---------------------------------------------------------------------------
# FD Jacobian comparison via torch.autograd.functional.jacobian
# ---------------------------------------------------------------------------


def test_fd_jacobian_per_grain_linear_chain() -> None:
    """Compute the Jacobian of a per-grain linear chain two ways and
    require they match: (a) ``torch.autograd.functional.jacobian`` over
    the raw torch function; (b) the v2-parity chain rule applied to the
    same chain through typed wrappers + the canonical seed.

    Chain: ``y[g] = 2.0 * x[g]`` for each grain g. Jacobian is
    ``2.0 * I_{Ng}``.
    """
    from neml2.es._helpers import _expanded_identity_seed
    from neml2.types import Scalar
    from neml2.types.functions import fullify

    N_grain = 4
    x_data = torch.linspace(0.1, 0.4, N_grain, dtype=torch.float64)

    # (a) FD Jacobian via autograd.
    def f(x):
        return 2.0 * x

    J_fd = torch.autograd.functional.jacobian(f, x_data)
    assert J_fd.shape == (N_grain, N_grain)
    assert torch.allclose(J_fd, 2.0 * torch.eye(N_grain, dtype=torch.float64))

    # (b) v2-parity chain rule: seed → 2 * tangent → fullify to get the
    # full (K, sub) Jacobian back.
    x = Scalar(x_data, sub_batch_ndim=1)
    seed = _expanded_identity_seed(x)
    # Compute tangent of f: y_tangent = 2.0 * seed.
    y_tan = seed * 2.0
    # Materialise to recover the full Jacobian matrix shape (K, sub).
    J_v2p = fullify(y_tan).data
    assert J_v2p.shape == (N_grain, N_grain)
    assert torch.allclose(J_v2p, J_fd)


def test_fd_jacobian_mean_reduction() -> None:
    """Per-grain mean: ``y = mean_g(x[g])``. Jacobian is the row vector
    ``[1/N, 1/N, ..., 1/N]`` of shape (1, N_grain). The chain rule
    must exit the sub axis via mean-expose and end with K_full of size N
    holding 1/N at every row."""
    from neml2.es._helpers import _expanded_identity_seed
    from neml2.types import Scalar
    from neml2.types.functions import mean

    N_grain = 5
    x_data = torch.linspace(0.1, 0.5, N_grain, dtype=torch.float64)

    def f(x):
        return x.mean()

    J_fd = torch.autograd.functional.jacobian(f, x_data)
    # autograd returns shape (N_grain,) for scalar→vec
    assert J_fd.shape == (N_grain,)
    assert torch.allclose(J_fd, torch.full((N_grain,), 1.0 / N_grain, dtype=torch.float64))

    x = Scalar(x_data.clone(), sub_batch_ndim=1)
    seed = _expanded_identity_seed(x)
    # Apply the mean reduction to the tangent: should expose K to size N.
    y_tan = mean(seed.sub_batch, dim=0)
    assert y_tan.k_ndim == 1
    assert y_tan.k_state == ("full",)
    assert y_tan.k_pairing == (None,)
    # K axis now holds the gradient w.r.t. each grain.
    assert y_tan.data.shape == (N_grain,)
    assert torch.allclose(y_tan.data, J_fd)
