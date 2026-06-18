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

"""Unit coverage for the equation-system raw-tensor boundary helpers.

``wrap_group_raw`` (``neml2.es.assembled``) and ``build_identity_seed``
(``neml2.es._helpers``) are the two seams where per-group raw tensors and
chain-rule seeds cross the typed boundary shared by the native eager Newton
path and the AOTI export wrappers. The end-to-end ``tests/aoti`` suite drives
them only under uniform-batch tracing, which never exercises the empty-group /
block wrapping or the dynamic-batch left-pad -- exactly the branches that
caused the multi-unknown chain-rule bug. These direct tests pin each branch.
"""

from __future__ import annotations

import torch

from neml2.es._helpers import build_identity_seed
from neml2.es.assembled import wrap_group_raw
from neml2.es.axis_layout import AxisLayout
from neml2.types import SR2, Scalar, Tensor

# ---------- wrap_group_raw: the three structural paths ----------


def test_wrap_group_raw_empty_group_passes_through():
    """An empty group carries no vars; the whole tensor is batch, no sub_batch."""
    raw = torch.zeros(4, 5)
    result = wrap_group_raw(raw, (), "dense", AxisLayout([[]], {}))
    assert isinstance(result, Tensor)
    assert result.batch_ndim == raw.ndim
    assert result.sub_batch_ndim == 0


def test_wrap_group_raw_dense_folds_sub_batch_into_base():
    """DENSE: shape ``(*dyn, group_total)`` -- a single trailing base axis,
    sub_batch folded in, so ``sub_batch_ndim == 0``."""
    raw = torch.zeros(8, 6)  # (batch, group_total)
    layout = AxisLayout([["x"]], {"x": SR2}, structure=["dense"])
    result = wrap_group_raw(raw, ("x",), "dense", layout)
    assert result.batch_ndim == raw.ndim - 1
    assert result.sub_batch_ndim == 0


def test_wrap_group_raw_block_preserves_sub_batch_axes():
    """BLOCK: shape ``(*dyn, *sub_batch, group_base_total)`` with ``base_ndim==1``;
    the sub_batch axes survive as ``sub_batch_ndim``."""
    raw = torch.zeros(8, 12, 6)  # (batch, sub_batch=12, group_base_total)
    layout = AxisLayout(
        [["x"]],
        {"x": SR2},
        sub_batch_shapes={"x": torch.Size([12])},
        structure=["block"],
    )
    result = wrap_group_raw(raw, ("x",), "block", layout)
    assert result.sub_batch_ndim == 1
    assert result.batch_ndim == raw.ndim - 1 - 1  # minus one sub axis, one base axis


# ---------- build_identity_seed: the dynamic-batch left-pad ----------


def test_build_identity_seed_left_pads_lower_dyn_ndim_variables():
    """Variables with fewer dynamic-batch axes than the system maximum are
    left-padded with size-1 axes so the leading-K tangent contract lines up
    position-wise. This is the path the AOTI wrapper previously omitted (it was
    invisible under uniform-batch tracing)."""
    batched = Scalar(torch.zeros(4))  # dyn_ndim = 1
    unbatched = Scalar(torch.zeros(()))  # dyn_ndim = 0 -> must be padded to 1
    state = {"a": batched, "b": unbatched}
    spec = {"a": Scalar, "b": Scalar}
    sub_batch_shapes = {"a": torch.Size(()), "b": torch.Size(())}

    seed = build_identity_seed(state, ["a", "b"], 1, spec, sub_batch_shapes)

    assert set(seed) == {"a", "b"}
    assert set(seed["a"]) == {"a:rgroup0"}
    assert set(seed["b"]) == {"b:rgroup0"}
    # The batched variable keeps its dyn extent; the unbatched one is padded
    # up to the same dynamic-batch rank with a leading size-1 axis.
    assert tuple(seed["a"]["a:rgroup0"].dynamic_batch_shape) == (4,)
    assert tuple(seed["b"]["b:rgroup0"].dynamic_batch_shape) == (1,)


def test_build_identity_seed_multiple_residual_groups():
    """One seed leaf per (input, residual_group); keys use the
    ``{name}:rgroup{gi}`` format the block-matrix builder reads."""
    state = {"a": Scalar(torch.zeros(4))}
    spec = {"a": Scalar}
    seed = build_identity_seed(state, ["a"], 3, spec, {"a": torch.Size(())})
    assert set(seed["a"]) == {"a:rgroup0", "a:rgroup1", "a:rgroup2"}


def test_build_identity_seed_empty_seed_names_is_empty():
    """No seeds requested -> empty mapping (the caller passes this when the
    system needs no chain-rule sensitivities)."""
    state = {"a": Scalar(torch.zeros(4))}
    seed = build_identity_seed(state, [], 1, {"a": Scalar}, {"a": torch.Size(())})
    assert seed == {}


def test_build_identity_seed_empty_state_is_empty():
    """Degenerate empty system: both the max-dyn-ndim scan and the seed loop
    iterate over nothing and the result is an empty mapping."""
    assert build_identity_seed({}, [], 1, {}, {}) == {}
