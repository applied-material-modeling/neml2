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

"""Structural tests for the per-sub-batch-axis broadcast-state fields.

Commit 1 of the chain-rule fast-path roll-out adds two metadata fields
to every typed-tensor wrapper. This module verifies the data model in
isolation — op-level state propagation is exercised in
``test_broadcast_invariance`` once Commit 2 lands the dispatch.
"""

from __future__ import annotations

import torch

from neml2.types import SR2, Scalar


def test_default_state_is_empty_tuple() -> None:
    """Legacy construction yields empty tuples — backwards-compat default."""
    s = Scalar(torch.tensor([1.0, 2.0]))
    assert s.sub_batch_state == ()
    assert s.sub_batch_meta == ()
    sr = SR2.fill(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
    assert sr.sub_batch_state == ()
    assert sr.sub_batch_meta == ()


def test_sub_batch_shape_reports_logical_extent_for_broadcast_axes() -> None:
    """When an axis is ``"broadcast"`` (size 1 in data), the property still
    returns the LOGICAL extent recorded in ``sub_batch_meta``."""
    bc = Scalar(
        torch.zeros(4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(12,),
    )
    assert bc.data.shape == (4, 1)
    assert bc.sub_batch_shape == torch.Size([12])


def test_sub_batch_shape_reads_data_for_full_axes() -> None:
    """Legacy / all-``"full"`` wrappers report data.shape for sub_batch_shape."""
    full = Scalar(torch.zeros(4, 12), sub_batch_ndim=1)
    assert full.sub_batch_shape == torch.Size([12])
    # Explicit "full" state with mismatched meta — meta is ignored when state="full"
    full2 = Scalar(
        torch.zeros(4, 12),
        sub_batch_ndim=1,
        sub_batch_state=("full",),
        sub_batch_meta=(999,),
    )
    assert full2.sub_batch_shape == torch.Size([12])


def test_materialize_inflates_broadcast_axes_to_logical_size() -> None:
    """``materialize()`` expands every broadcast axis to its logical extent
    and clears the per-axis flags to all-``"full"``."""
    bc = Scalar(
        torch.arange(4.0).unsqueeze(-1),  # shape (4, 1)
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(12,),
    )
    mat = bc.materialize()
    assert mat.data.shape == (4, 12)
    assert mat.sub_batch_state == ("full",)
    assert mat.sub_batch_meta == (12,)
    # Every slot of the materialized axis carries the same broadcast value
    assert torch.allclose(mat.data[:, 0], mat.data[:, 11])


def test_materialize_is_noop_on_all_full() -> None:
    """No-op when state is empty or every axis is already ``"full"``."""
    s = Scalar(torch.zeros(4, 12), sub_batch_ndim=1)
    assert s.materialize() is s
    s2 = Scalar(
        torch.zeros(4, 12),
        sub_batch_ndim=1,
        sub_batch_state=("full",),
        sub_batch_meta=(12,),
    )
    assert s2.materialize() is s2


def test_rewrap_preserves_state_when_ndim_unchanged() -> None:
    """``_rewrap`` carries state through ops that don't change sub_batch_ndim."""
    bc = Scalar(
        torch.zeros(4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(12,),
    )
    # Same ndim, new data — state must carry through
    rewrapped = bc._rewrap(torch.ones(4, 1), sub_batch_ndim=1)
    assert rewrapped.sub_batch_state == ("broadcast",)
    assert rewrapped.sub_batch_meta == (12,)


def test_sub_batch_unsqueeze_preserves_state() -> None:
    """``SubBatchView.unsqueeze`` keeps non-empty state across the
    sub_batch_ndim bump, marking the new axes ``"broadcast"`` with meta=1
    as placeholders. A subsequent op against a sized primal picks up the
    real meta via :func:`combine_sub_batch_state`."""
    bc = Scalar(
        torch.zeros(4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(12,),
    )
    out = bc.sub_batch.unsqueeze(-1)
    assert out.sub_batch_ndim == 2
    assert out.sub_batch_state == ("broadcast", "broadcast")
    assert out.sub_batch_meta == (12, 1)
    # Inserting at the front position:
    out_front = bc.sub_batch.unsqueeze(0)
    assert out_front.sub_batch_state == ("broadcast", "broadcast")
    assert out_front.sub_batch_meta == (1, 12)


def test_sub_batch_unsqueeze_legacy_stays_legacy() -> None:
    """Empty-state wrappers (legacy lane) stay empty after unsqueeze — the
    state override only kicks in when state is non-empty."""
    legacy = Scalar(torch.zeros(4, 12), sub_batch_ndim=1)
    out = legacy.sub_batch.unsqueeze(-1)
    assert out.sub_batch_ndim == 2
    assert out.sub_batch_state == ()
    assert out.sub_batch_meta == ()


def test_sub_batch_squeeze_drops_state_axis() -> None:
    """``SubBatchView.squeeze`` removes the matching state/meta entry."""
    bc = Scalar(
        torch.zeros(4, 12, 1),
        sub_batch_ndim=2,
        sub_batch_state=("full", "broadcast"),
        sub_batch_meta=(12, 7),
    )
    out = bc.sub_batch.squeeze(-1)
    assert out.sub_batch_ndim == 1
    assert out.sub_batch_state == ("full",)
    assert out.sub_batch_meta == (12,)


def test_rewrap_clears_state_when_ndim_changes() -> None:
    """``_rewrap`` falls back to all-``"full"`` (empty tuple) when ndim shifts —
    forces explicit handling at axis-adding/removing ops."""
    bc = Scalar(
        torch.zeros(4, 1),
        sub_batch_ndim=1,
        sub_batch_state=("broadcast",),
        sub_batch_meta=(12,),
    )
    rewrapped = bc._rewrap(torch.zeros(4), sub_batch_ndim=0)
    assert rewrapped.sub_batch_state == ()
    assert rewrapped.sub_batch_meta == ()
