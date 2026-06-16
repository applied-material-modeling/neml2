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

"""V2P-1 pre-tests: pin the multi-axis K-region contract on TensorWrapper.

These tests pin the surface specified in the v2-parity-chain-rule plan
(Appendix A: field surface; Appendix B: function primitives). They are
RED on clean HEAD — the symbols (``k_ndim``, ``k_state``, ``k_pairing``,
``align_k``, ``combine_k_state``, ``fullify``, ``sum_sub_batch``)
don't exist yet. They turn green incrementally as V2P-2 through V2P-4
land.

Test imports happen inside each test body so a single missing symbol
fails just that test (collection error → red per-test), not the whole
module.
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Field surface (Appendix A)
# ---------------------------------------------------------------------------


def test_default_k_fields_are_empty() -> None:
    """Default construction of a wrapper carries no K region."""
    from neml2.types import Scalar

    s = Scalar(torch.tensor([1.0, 2.0]))
    assert s.k_ndim == 0
    assert s.k_state == ()
    assert s.k_pairing == ()


def test_k_fields_round_trip_through_constructor() -> None:
    """Explicit K fields seat through the dataclass init unchanged."""
    from neml2.types import Scalar

    # Per-grain seed for a Scalar input: K_paired axis, no base dir.
    s = Scalar(
        torch.zeros(1, 4, 1),  # (K_grain=1, dyn=4, N_grain=1; broadcast at sub axis)
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(7,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    assert s.k_ndim == 1
    assert s.k_state == ("broadcast",)
    assert s.k_pairing == (0,)


def test_k_state_length_must_equal_k_ndim() -> None:
    """``len(k_state)`` must equal ``k_ndim``; mismatch raises."""
    from neml2.types import Scalar

    with pytest.raises((ValueError, AssertionError)):
        Scalar(
            torch.zeros(1, 4),
            sub_batch_ndim=0,
            k_ndim=1,
            k_state=(),  # wrong length
            k_pairing=(None,),
        )


def test_k_pairing_length_must_equal_k_ndim() -> None:
    """``len(k_pairing)`` must equal ``k_ndim``; mismatch raises."""
    from neml2.types import Scalar

    with pytest.raises((ValueError, AssertionError)):
        Scalar(
            torch.zeros(1, 4),
            sub_batch_ndim=0,
            k_ndim=1,
            k_state=("full",),
            k_pairing=(),  # wrong length
        )


def test_k_pairing_must_reference_valid_sub_batch_axis() -> None:
    """A non-None ``k_pairing[i]`` must be in ``[0, sub_batch_ndim)``."""
    from neml2.types import Scalar

    with pytest.raises((ValueError, IndexError)):
        Scalar(
            torch.zeros(1, 4),
            sub_batch_ndim=1,
            k_ndim=1,
            k_state=("broadcast",),
            k_pairing=(5,),  # out of range
        )


def test_k_ndim_plus_regions_match_data_ndim() -> None:
    """``data.ndim == k_ndim + dyn_ndim + sub_batch_ndim + BASE_NDIM``."""
    from neml2.types import SR2

    # SR2 base_ndim=1, sub_batch_ndim=1, k_ndim=2 (K_paired + K_base),
    # dyn=1 → data.ndim = 2 + 1 + 1 + 1 = 5
    s = SR2(
        torch.zeros(1, 6, 4, 1, 6),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(7,),
        k_ndim=2,
        k_state=("broadcast", "full"),
        k_pairing=(0, None),
    )
    assert s.data.ndim == s.k_ndim + 1 + s.sub_batch_ndim + SR2.BASE_NDIM


def test_paired_broadcast_invariant_size_one() -> None:
    """A K_paired ``"broadcast"`` axis must have ``data.shape`` size 1
    at the K axis position (the paired-broadcast invariant)."""
    from neml2.types import Scalar

    with pytest.raises((ValueError, AssertionError)):
        Scalar(
            torch.zeros(3, 4, 1),  # K axis size 3 but state="broadcast" + pairing
            sub_batch_ndim=1,
            sub_batch_state=("broadcast",),
            sub_batch_meta=(7,),
            k_ndim=1,
            k_state=("broadcast",),
            k_pairing=(0,),  # paired-broadcast must be size 1 in data
        )


# ---------------------------------------------------------------------------
# _rewrap K threading (Appendix A)
# ---------------------------------------------------------------------------


def test_rewrap_preserves_k_state_when_k_ndim_unchanged() -> None:
    """``_rewrap`` carries K state through ops that don't change k_ndim."""
    from neml2.types import Scalar

    s = Scalar(
        torch.zeros(1, 4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(7,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    rewrapped = s._rewrap(torch.ones(1, 4, 1), sub_batch_ndim=1)
    assert rewrapped.k_ndim == 1
    assert rewrapped.k_state == ("broadcast",)
    assert rewrapped.k_pairing == (0,)


def test_rewrap_requires_explicit_k_when_k_ndim_changes() -> None:
    """``_rewrap`` with a new k_ndim either takes explicit K state/pairing
    or resets to empty — caller can't accidentally inherit stale state
    across a K-rank change."""
    from neml2.types import Scalar

    s = Scalar(
        torch.zeros(1, 4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(7,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    # Explicit-empty K is allowed (e.g. for a primal-shaped rewrap).
    out = s._rewrap(
        torch.zeros(4, 7),
        sub_batch_ndim=1,
        k_ndim=0,
        k_state=(),
        k_pairing=(),
    )
    assert out.k_ndim == 0
    assert out.k_state == ()
    assert out.k_pairing == ()


# ---------------------------------------------------------------------------
# align_k helper (Appendix A)
# ---------------------------------------------------------------------------


def test_align_k_pads_lower_k_ndim_to_max() -> None:
    """align_k inserts size-1 broadcast K axes at the leading positions
    of lower-k_ndim operands so they match the max."""
    from neml2.types import Scalar
    from neml2.types._base import align_k

    # a: k_ndim=2  (K_paired + K_base sketch)
    a = Scalar(
        torch.zeros(1, 6, 4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(7,),
        k_ndim=2,
        k_state=("broadcast", "full"),
        k_pairing=(0, None),
    )
    # b: k_ndim=0 (primal-shaped) — needs two padded leading K axes
    b = Scalar(
        torch.zeros(4, 7),
        sub_batch_ndim=1,
        sub_batch_state=("full",),
        sub_batch_meta=(7,),
    )
    (aa, bb), kmax = align_k(a, b)
    assert kmax == 2
    assert aa is a  # already at max — identity return
    assert bb.k_ndim == 2
    assert bb.k_state == ("broadcast", "broadcast")  # padded axes default broadcast
    assert bb.k_pairing == (None, None)  # padded axes default unpaired


def test_align_k_no_op_when_all_equal() -> None:
    """align_k returns inputs unchanged when every operand has the same k_ndim."""
    from neml2.types import Scalar
    from neml2.types._base import align_k

    a = Scalar(torch.zeros(6, 4), sub_batch_ndim=0, k_ndim=1, k_state=("full",), k_pairing=(None,))
    b = Scalar(torch.zeros(6, 4), sub_batch_ndim=0, k_ndim=1, k_state=("full",), k_pairing=(None,))
    (aa, bb), kmax = align_k(a, b)
    assert kmax == 1
    assert aa is a
    assert bb is b


# ---------------------------------------------------------------------------
# combine_k_state lattice (Appendix A)
# ---------------------------------------------------------------------------


def test_combine_k_state_any_full_wins() -> None:
    """If any operand is ``"full"`` at axis i, result axis i is ``"full"``."""
    from neml2.types import Scalar
    from neml2.types._base import combine_k_state

    a = Scalar(
        torch.zeros(6, 4, 7),
        sub_batch_ndim=1,
        k_ndim=1,
        k_state=("full",),
        k_pairing=(None,),
    )
    b = Scalar(
        torch.zeros(1, 4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(7,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    state, pairing = combine_k_state(a, b)
    assert state == ("full",)
    # When mixing full + paired-broadcast, the pairing is broken because the
    # full lane has no per-site interpretation. Result pairing is None.
    assert pairing == (None,)


def test_combine_k_state_all_broadcast_stays_broadcast() -> None:
    """Result axis is ``"broadcast"`` when every operand is ``"broadcast"``."""
    from neml2.types import Scalar
    from neml2.types._base import combine_k_state

    a = Scalar(
        torch.zeros(1, 4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(7,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    b = Scalar(
        torch.zeros(1, 4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(7,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    state, pairing = combine_k_state(a, b)
    assert state == ("broadcast",)
    assert pairing == (0,)


def test_combine_k_state_disagreeing_pairings_raise() -> None:
    """Two operands with broadcast K paired to DIFFERENT sub axes raise —
    the chain rule should have aligned them upstream."""
    from neml2.types import Scalar
    from neml2.types._base import combine_k_state

    a = Scalar(
        torch.zeros(1, 4, 1, 5),
        sub_batch_ndim=2,
        sub_batch_state=("broadcast", "full"),
        sub_batch_meta=(7, 5),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    b = Scalar(
        torch.zeros(1, 4, 7, 1),
        sub_batch_ndim=2,
        sub_batch_state=("full", "broadcast"),
        sub_batch_meta=(7, 5),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(1,),
    )
    with pytest.raises(ValueError):
        combine_k_state(a, b)


# ---------------------------------------------------------------------------
# fullify primitive (Appendix B)
# ---------------------------------------------------------------------------


def test_fullify_materializes_paired_broadcast_to_full() -> None:
    """fullify expands every K-paired broadcast axis from size 1 to its
    paired sub axis's extent, eye-shaped on the (K_i, sub_pair) diagonal."""
    from neml2.types import Scalar
    from neml2.types.functions import fullify

    # K_grain=1 broadcast paired with sub axis 0 (size 4).
    # data values are constant 1.0; after fullify, value at (K=k, ..., sub=g) is
    # 1.0 * δ(k==g).
    s = Scalar(
        torch.ones(1, 4),  # (K=1, sub=4)  -- here dyn is empty
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(4,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    out = fullify(s)
    assert out.k_ndim == 1
    assert out.k_state == ("full",)
    assert out.k_pairing == (None,)
    # Result shape: (K=4, sub=4); eye on (K, sub) diagonal.
    assert out.data.shape == (4, 4)
    expected = torch.eye(4, dtype=out.data.dtype)
    assert torch.allclose(out.data, expected)


def test_fullify_no_op_when_no_paired_broadcast() -> None:
    """fullify is a no-op when no K axis is paired-broadcast."""
    from neml2.types import Scalar
    from neml2.types.functions import fullify

    # K full, no pairing.
    s = Scalar(
        torch.ones(6, 4),
        sub_batch_ndim=0,
        k_ndim=1,
        k_state=("full",),
        k_pairing=(None,),
    )
    out = fullify(s)
    assert out is s  # exact identity, no copy


# ---------------------------------------------------------------------------
# sum_sub_batch primitive (Appendix B)
# ---------------------------------------------------------------------------


def test_sum_sub_batch_axis_not_k_paired_is_regular_sum() -> None:
    """When the reduce axis has no K pairing, sum_sub_batch is a plain sum
    that drops the sub axis."""
    from neml2.types import Tensor as DynTensor
    from neml2.types.functions import sum_sub_batch

    # Primal-shaped Tensor with sub_batch_ndim=1, no K.
    t = DynTensor(
        torch.arange(8.0).reshape(2, 4),  # (dyn=2, sub=4)
        batch_ndim=1,
        sub_batch_ndim=1,
    )
    out = sum_sub_batch(t, axis=0)
    assert out.sub_batch_ndim == 0
    assert out.data.shape == (2,)
    assert torch.allclose(out.data, torch.tensor([6.0, 22.0]))


def test_sum_sub_batch_axis_k_paired_exposes_k() -> None:
    """When the reduce axis has a K paired with it (broadcast), sum_sub_batch
    promotes that K axis from broadcast to full (size = sub extent), drops
    the K pairing, and drops the sub axis."""
    from neml2.types import Tensor as DynTensor
    from neml2.types.functions import sum_sub_batch

    # data: (K=1, dyn=2, sub=4) with K_paired broadcast on sub axis 0.
    # After expose: result has (K=4, dyn=2); K axis carries the per-site
    # original tangent value (no actual summation — the diagonal entries
    # promote, off-diagonals would have been zero).
    t = DynTensor(
        torch.ones(1, 2, 4),
        batch_ndim=1,
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(4,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    out = sum_sub_batch(t, axis=0)
    assert out.sub_batch_ndim == 0
    assert out.k_ndim == 1
    assert out.k_state == ("full",)
    assert out.k_pairing == (None,)
    # Shape: (K=4, dyn=2). Each K row carries the per-site value.
    assert out.data.shape == (4, 2)


# ---------------------------------------------------------------------------
# sum / mean expose semantics (Appendix B cheat-sheet)
# ---------------------------------------------------------------------------


def test_sum_view_exposes_k_on_paired_reduce_axis() -> None:
    """``sum(t.sub_batch, dims=0)`` exposes K when that axis is K-paired-broadcast."""
    from neml2.types import Scalar
    from neml2.types.functions import sum as ts_sum

    s = Scalar(
        torch.ones(1, 4),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(4,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    out = ts_sum(s.sub_batch, dims=0)
    assert out.sub_batch_ndim == 0
    assert out.k_ndim == 1
    assert out.k_state == ("full",)
    assert out.k_pairing == (None,)
    # K=4 rows of the per-site value (no /N scaling for sum).
    assert out.data.shape == (4,)
    assert torch.allclose(out.data, torch.ones(4))


def test_mean_view_exposes_k_and_divides_by_n() -> None:
    """``mean(t.sub_batch, dim=0)`` exposes K AND divides by N (the reduce
    axis extent) — per the Appendix B cheat-sheet."""
    from neml2.types import Scalar
    from neml2.types.functions import mean

    s = Scalar(
        torch.ones(1, 4),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(4,),
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    out = mean(s.sub_batch, dim=0)
    assert out.sub_batch_ndim == 0
    assert out.k_ndim == 1
    assert out.k_state == ("full",)
    assert out.k_pairing == (None,)
    assert out.data.shape == (4,)
    assert torch.allclose(out.data, torch.full((4,), 0.25))
