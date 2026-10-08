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

r"""Unit coverage for the (common x extra) partial-fold assembler API.

The equation-system assembler preserves a BLOCK group's shared leading sub-batch
prefix (e.g. a per-common-site axis) as the block intermediate and folds any EXTRA
trailing sub-batch axis (e.g. a per-extra-index axis) into each variable's per-site base,
unfolding it on disassembly. These tests pin that machinery directly -- the new
``AxisLayout.group_common_sub_batch`` / ``var_extra_sub_batch`` API, the
``AssembledVector.from_dict`` -> ``disassemble`` fold round-trip, and the
group-common tagging in ``wrap_group_raw`` -- rather than only through a full
solve. The round-trip uses distinct per-(common, extra, component) values so a
mis-ordered fold (extra <-> base transpose, or non-extra-major) is caught: the
assembled block's ``data.reshape`` on both the vector side here and the dense
tangent side (``_convert_tangent_to_paired_block``'s ``col_storage`` branch)
assume the SAME ``[extra_outer, base_inner]`` layout, so pinning it here guards
both.
"""

from __future__ import annotations

import pytest
import torch

from neml2.es.assembled import AssembledMatrix, AssembledVector, wrap_group_raw
from neml2.es.axis_layout import AxisLayout
from neml2.types import SR2, Scalar, TensorWrapper


def _common_extra_layout(extra_cls, n_common: int, n_extra: int) -> AxisLayout:
    """Block group {u_common:(common,), u_extra:(common, extra)} -- shared prefix + folded axis."""
    return AxisLayout(
        [["u_common", "u_extra"]],
        {"u_common": Scalar, "u_extra": extra_cls},
        sub_batch_shapes={
            "u_common": torch.Size([n_common]),
            "u_extra": torch.Size([n_common, n_extra]),
        },
        structure=["block"],
    )


def test_from_dict_disassemble_roundtrip_nc2_extra2_unequal():
    """Multi-axis generality: two common axes AND two folded extra axes, all of
    distinct extent, round-trip exactly.

    The shipped case is nc=1 (common) / extra=1 (extra). The fold is coded generically
    in the axis counts, so this pins nc=2 / extra=2 directly. Extents are all
    different (common (2, 3), extra (4, 5)) and every element value is distinct, so a
    swap of either the two common axes or the two extra axes -- which an equal-extent
    case could hide -- scrambles the round-trip and fails.
    """
    g1, g2, s1, s2 = 2, 3, 4, 5
    layout = AxisLayout(
        [["u_common", "u_extra"]],
        {"u_common": Scalar, "u_extra": Scalar},
        sub_batch_shapes={
            "u_common": torch.Size([g1, g2]),
            "u_extra": torch.Size([g1, g2, s1, s2]),
        },
        structure=["block"],
    )
    assert tuple(layout.group_common_sub_batch(0)) == (g1, g2)
    assert tuple(layout.var_extra_sub_batch(0, "u_extra")) == (s1, s2)
    # Distinct value per (g1, g2, s1, s2) element.
    extra_vals = torch.arange(g1 * g2 * s1 * s2, dtype=torch.float64).reshape(1, g1, g2, s1, s2)
    common_vals = torch.arange(g1 * g2, dtype=torch.float64).reshape(1, g1, g2)
    values = {
        "u_common": Scalar(common_vals, 1).with_sub_batch_ndim(2),
        "u_extra": Scalar(extra_vals, 1).with_sub_batch_ndim(4),
    }
    back = AssembledVector.from_dict(layout, values).disassemble().values
    torch.testing.assert_close(back["u_common"].data, values["u_common"].data, rtol=0, atol=0)
    torch.testing.assert_close(back["u_extra"].data, values["u_extra"].data, rtol=0, atol=0)


def test_group_common_and_extra_sub_batch():
    """common is the shared block intermediate; extra is the folded extra axis."""
    layout = _common_extra_layout(Scalar, n_common=2, n_extra=3)
    assert tuple(layout.group_common_sub_batch(0)) == (2,)
    assert tuple(layout.var_extra_sub_batch(0, "u_common")) == ()
    assert tuple(layout.var_extra_sub_batch(0, "u_extra")) == (3,)


def test_group_common_sub_batch_rejects_inconsistent_prefix():
    """A BLOCK group whose variables do not share a leading prefix is rejected."""
    layout = AxisLayout(
        [["a", "b"]],
        {"a": Scalar, "b": Scalar},
        sub_batch_shapes={"a": torch.Size([2, 3]), "b": torch.Size([5, 3])},
        structure=["block"],
    )
    try:
        layout.group_common_sub_batch(0)
    except ValueError as e:
        assert "common" in str(e).lower()
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a ValueError for an inconsistent block prefix")


def test_from_dict_disassemble_roundtrip_scalar():
    """Scalar (base=1) per (common, extra): the fold then unfold recovers every value.

    base=1 is the common production case (a scalar per-site state); distinct values
    pin that the extra axis is folded and restored in order.
    """
    n_common, n_extra = 2, 3
    layout = _common_extra_layout(Scalar, n_common, n_extra)
    # u_extra[common, extra] = common*10 + extra -- all distinct.
    common = torch.arange(n_common, dtype=torch.float64)
    extra = torch.arange(n_extra, dtype=torch.float64)
    u_extra_vals = (common.reshape(n_common, 1) * 10 + extra.reshape(1, n_extra)).reshape(
        1, n_common, n_extra
    )
    u_common_vals = common.reshape(1, n_common)
    values = {
        "u_common": Scalar(u_common_vals, 1).with_sub_batch_ndim(1),
        "u_extra": Scalar(u_extra_vals, 1).with_sub_batch_ndim(2),
    }
    back = AssembledVector.from_dict(layout, values).disassemble().values
    torch.testing.assert_close(back["u_common"].data, values["u_common"].data, rtol=0, atol=0)
    torch.testing.assert_close(back["u_extra"].data, values["u_extra"].data, rtol=0, atol=0)


def test_from_dict_disassemble_roundtrip_sr2_pins_extra_base_order():
    """SR2 (base=6) per (common, extra): the fold must keep [extra_outer, base_inner].

    With base > 1 a extra <-> base transpose in the fold is observable (unlike the
    base=1 Scalar case). Every (common, extra, component) entry is distinct, so any
    re-ordering of the folded ``extra * base`` storage corrupts the round-trip.
    """
    n_common, n_extra, nbase = 2, 3, 6
    layout = _common_extra_layout(SR2, n_common, n_extra)
    # Distinct value per (common, extra, component).
    g = torch.arange(n_common, dtype=torch.float64).reshape(n_common, 1, 1)
    s = torch.arange(n_extra, dtype=torch.float64).reshape(1, n_extra, 1)
    c = torch.arange(nbase, dtype=torch.float64).reshape(1, 1, nbase)
    u_extra_vals = (g * 100 + s * 10 + c).reshape(1, n_common, n_extra, nbase)
    values = {
        "u_common": Scalar(
            torch.arange(n_common, dtype=torch.float64).reshape(1, n_common), 1
        ).with_sub_batch_ndim(1),
        "u_extra": SR2(u_extra_vals, 1).with_sub_batch_ndim(2),
    }
    back = AssembledVector.from_dict(layout, values).disassemble().values
    torch.testing.assert_close(back["u_extra"].data, values["u_extra"].data, rtol=0, atol=0)


def test_wrap_group_raw_two_sub_batch_block_tags_common_prefix():
    """A BLOCK group with a (common, extra) variable is wrapped with the group COMMON
    prefix (common) as sub_batch, not the variable's full (common, extra) -- the fix
    that makes the native ImplicitUpdate path fold-aware.

    The raw per-group tensor is ``(*dyn, common, group_base_total)``: the common axis
    stays an intermediate axis, while the extra axis is already folded into the base
    total, so the wrapped tensor has ``sub_batch_ndim == 1`` (common only).
    """
    n_common, n_extra = 2, 3
    layout = _common_extra_layout(Scalar, n_common, n_extra)
    # group base total = u_common(1) + u_extra(extra folded into base) = 1 + n_extra.
    group_base_total = 1 + n_extra
    raw = torch.zeros(4, n_common, group_base_total)  # (dyn, common, base_total)
    result = wrap_group_raw(raw, ("u_common", "u_extra"), "block", layout)
    assert result.sub_batch_ndim == 1  # common only; extra folded into base
    assert tuple(result.sub_batch_shape) == (n_common,)
    assert result.batch_ndim == raw.ndim - 1 - 1  # minus common sub axis, minus base axis


def test_group_common_override_retains_matched_prefix():
    """A lone (N, M) member infers (N, M) as its common, but an explicit override pins
    (N,) -- the generated-given-layout fix so disassembly keeps the folded M width.

    Without the override, ``group_common_sub_batch`` takes the member's full shape and
    ``var_extra_sub_batch`` is empty (folded width collapses to 1, dropping M). With it,
    the extra axis (M,) is preserved and the per-site storage is ``base * M``.
    """
    n, m = 4, 3
    specs: dict[str, type[TensorWrapper]] = {"g_extra": Scalar}
    shapes = {"g_extra": torch.Size([n, m])}
    inferred = AxisLayout([["g_extra"]], specs, shapes, structure=["block"])
    assert tuple(inferred.group_common_sub_batch(0)) == (n, m)  # the bug: infers full
    assert tuple(inferred.var_extra_sub_batch(0, "g_extra")) == ()

    pinned = AxisLayout(
        [["g_extra"]], specs, shapes, structure=["block"], group_common=[torch.Size([n])]
    )
    assert tuple(pinned.group_common_sub_batch(0)) == (n,)
    assert tuple(pinned.var_extra_sub_batch(0, "g_extra")) == (m,)
    # The storage width assembly/disassembly slice by must include the folded M.
    assert AssembledMatrix._var_storage(pinned, 0, "g_extra", "block") == m
    assert AssembledMatrix._var_storage(inferred, 0, "g_extra", "block") == 1


def test_group_common_override_rejects_non_prefix_member():
    """An override that a member does not begin with is rejected."""
    layout = AxisLayout(
        [["g"]],
        {"g": Scalar},
        {"g": torch.Size([4, 3])},
        structure=["block"],
        group_common=[torch.Size([5])],
    )
    with pytest.raises(ValueError, match="explicit common prefix"):
        layout.group_common_sub_batch(0)


def test_group_common_length_validated():
    """group_common must have one entry per group."""
    with pytest.raises(ValueError, match="one per group"):
        AxisLayout(
            [["a"], ["b"]],
            {"a": Scalar, "b": Scalar},
            structure=["block", "dense"],
            group_common=[torch.Size([2])],
        )


def test_fullify_sub_axis_selects_paired_k_and_resolves_negative_index():
    """``fullify(sub_axis=)`` materializes only the K paired with that sub axis.

    A wrong sub axis leaves the wrapper untouched; a negative index resolves to the
    paired axis and eye-expands K (broadcast size 1 -> full size N, diagonal on the
    (K, sub_pair) pair).
    """
    from neml2.types.functions import fullify

    n = 3
    w = Scalar(
        torch.arange(1.0, n + 1).reshape(1, n),  # K=1 (broadcast), sub=n
        sub_batch_ndim=1,
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    assert fullify(w, sub_axis=1) is w  # no K paired with sub axis 1 -> unchanged
    out = fullify(w, sub_axis=-1)  # resolves to sub axis 0
    assert out.k_state == ("full",)
    assert out.k_pairing == (None,)
    assert out.data.shape == (n, n)
    torch.testing.assert_close(out.data, torch.diag(torch.arange(1.0, n + 1)))


def test_fullify_noop_without_k_state():
    """A wrapper with no K region is returned unchanged."""
    from neml2.types.functions import fullify

    w = Scalar(torch.ones(2, 2), sub_batch_ndim=1)
    assert fullify(w, sub_axis=0) is w


def test_equalize_tangent_K_eye_expands_paired_broadcast():
    """A paired-broadcast K contribution is eye-expanded (not tiled) to the max K."""
    from neml2.models.chain_rule import equalize_tangent_K

    n = 3
    c_full = Scalar(
        torch.randn(n, n), sub_batch_ndim=1, k_ndim=1, k_state=("full",), k_pairing=(None,)
    )
    c_bcast = Scalar(
        torch.arange(1.0, n + 1).reshape(1, n),
        sub_batch_ndim=1,
        k_ndim=1,
        k_state=("broadcast",),
        k_pairing=(0,),
    )
    out = equalize_tangent_K([c_full, c_bcast])
    assert out[0].data.shape[0] == n
    assert out[1].data.shape[0] == n and out[1].k_state == ("full",)
    torch.testing.assert_close(out[1].data, torch.diag(torch.arange(1.0, n + 1)))


def test_equalize_tangent_K_tiles_compact_unpaired():
    """A compact (unpaired) full-K contribution is broadcast-tiled to the max K."""
    from neml2.models.chain_rule import equalize_tangent_K

    n = 3
    c_big = Scalar(
        torch.randn(n, 2), sub_batch_ndim=0, k_ndim=1, k_state=("full",), k_pairing=(None,)
    )
    c_small = Scalar(
        torch.ones(1, 2), sub_batch_ndim=0, k_ndim=1, k_state=("full",), k_pairing=(None,)
    )
    out = equalize_tangent_K([c_big, c_small])
    assert out[1].data.shape[0] == n
    torch.testing.assert_close(out[1].data, torch.ones(n, 2))
