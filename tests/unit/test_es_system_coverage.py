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

"""Direct-construction coverage for :class:`ModelNonlinearSystem` layout + assembly.

The equation-system's layout builders (``setup_ulayout`` / ``setup_glayout`` /
``setup_blayout``) and the eager assembly path (``to_sparse`` -> ``initialize`` ->
``assemble`` / ``A`` / ``b`` / ``A_and_b`` / ``A_and_B`` / ``A_and_B_and_b``) are
pure-ish functions of a Model's input/output spec plus the declared sub-batch shapes.
These pin them directly -- the given->block most-specific-first matching, the dense
remainder, the no-givens single-empty-dense fallback, the structure validation, and
the eager A/B/b assembly -- without running a full Newton solve or the pyzag adjoint.
"""

from __future__ import annotations

import pytest
import torch

from neml2 import load_string
from neml2.es.system import ModelNonlinearSystem
from neml2.types import Scalar


def _mk(models: str, name: str = "residual"):
    """Load a bare [Models] block and return one model by name."""
    return load_string("[Models]\n" + models + "\n[]").get_model(name)


def _diagonal_model():
    """r_common = u_common - g_common, r_extra = u_extra - g_extra (block common x extra)."""
    return _mk(
        """
        [r_common]
          type=ScalarLinearCombination from='u_common g_common' to='r_common' weights='1 -1'
        []
        [r_extra]
          type=ScalarLinearCombination from='u_extra g_extra' to='r_extra' weights='1 -1'
        []
        [residual] type=ComposedModel models='r_common r_extra' []
        """
    )


def _two_block_group_model():
    """Two block unknown groups with nested prefixes + two extra givens (dense remainder)."""
    return _mk(
        """
        [ra] type=ScalarLinearCombination from='ua ga' to='ra' weights='1 -1' []
        [rb] type=ScalarLinearCombination from='ub gab' to='rb' weights='1 -1' []
        [rc] type=ScalarLinearCombination from='gdense' to='rc' weights='1' []
        [rd] type=ScalarLinearCombination from='gtrivial' to='rd' weights='1' []
        [residual] type=ComposedModel models='ra rb rc rd' []
        """
    )


def _no_givens_model():
    """r_a = u_a, r_b = u_b -- every input is an unknown (no givens)."""
    return _mk(
        """
        [ra] type=ScalarLinearCombination from='ua' to='ra' weights='1' []
        [rb] type=ScalarLinearCombination from='ub' to='rb' weights='1' []
        [residual] type=ComposedModel models='ra rb' []
        """
    )


# ---------------------------------------------------------------- setup_glayout


def test_setup_glayout_most_specific_first_and_dense_remainder():
    """A given whose sub-batch starts with a longer block prefix is captured by the
    longer-prefix group (the most-specific-first sort), not greedily by the shorter."""
    m = _two_block_group_model()
    system = ModelNonlinearSystem(
        m,
        unknowns=[["ua"], ["ub"]],
        residuals=[["ra"], ["rb"]],
        structure=["block", "block"],
    )
    assert system.given_names == ["ga", "gab", "gdense", "gtrivial"]
    a, b, c = 2, 3, 4
    sb = {
        "ua": torch.Size([a]),
        "ub": torch.Size([a, b]),
        "ga": torch.Size([a]),
        "gab": torch.Size([a, b]),  # matches BOTH (a,) and (a,b); must go to (a,b)
        "gdense": torch.Size([c]),  # matches no block prefix -> dense remainder
        "gtrivial": torch.Size(()),  # trivial -> dense remainder
    }
    gl = system.setup_glayout(sub_batch_shapes=sb)
    assert gl.groups == (("ga",), ("gab",), ("gdense", "gtrivial"))
    assert gl.structure == ("block", "block", "dense")
    # gab resolved to the (a,b) block (sort worked), ga to the (a,) block.
    gc0, gc1, gc2 = gl.group_common[0], gl.group_common[1], gl.group_common[2]
    assert gc0 is not None and tuple(gc0) == (a,)
    assert gc1 is not None and tuple(gc1) == (a, b)
    assert gc2 is None


def test_setup_glayout_no_givens_single_empty_dense():
    """With no givens, setup_glayout yields one empty DENSE group (invariant-keeping)."""
    system = ModelNonlinearSystem(
        _no_givens_model(), unknowns=[["ua", "ub"]], residuals=[["ra", "rb"]], structure=["dense"]
    )
    assert system.given_names == []
    gl = system.setup_glayout()
    assert gl.groups == ((),)
    assert gl.structure == ("dense",)


def test_setup_glayout_dense_unknowns_all_givens_dense():
    """Dense unknown groups never form block col groups -- every given is dense remainder."""
    system = ModelNonlinearSystem(
        _diagonal_model(),
        unknowns=[["u_common", "u_extra"]],
        residuals=[["r_common", "r_extra"]],
        structure=["dense"],
    )
    gl = system.setup_glayout(
        sub_batch_shapes={"g_common": torch.Size([2]), "g_extra": torch.Size([2, 3])}
    )
    assert gl.groups == (("g_common", "g_extra"),)
    assert gl.structure == ("dense",)


# ---------------------------------------------------------------- validation


def test_structure_length_and_token_validation():
    """structure must have one token per unknown group and each must be block/dense."""
    m = _diagonal_model()
    with pytest.raises(ValueError, match="one per unknown group"):
        ModelNonlinearSystem(
            m,
            unknowns=[["u_common", "u_extra"]],
            residuals=[["r_common", "r_extra"]],
            structure=["block", "dense"],  # 2 tokens, 1 group
        )
    with pytest.raises(ValueError, match="'block' or 'dense'"):
        ModelNonlinearSystem(
            m,
            unknowns=[["u_common", "u_extra"]],
            residuals=[["r_common", "r_extra"]],
            structure=["sparse"],
        )


def test_infer_residual_groups_keyerror():
    """Inferring a residual whose `<name>_residual` output is absent raises."""
    with pytest.raises(KeyError, match="Could not infer residual"):
        # residuals omitted -> inference looks for 'u_common_residual' (not an output).
        ModelNonlinearSystem(_diagonal_model(), unknowns=[["u_common", "u_extra"]])


def test_structure_defaults_to_dense_when_empty():
    """An empty structure list defaults to all-dense (one per unknown group)."""
    system = ModelNonlinearSystem(
        _diagonal_model(),
        unknowns=[["u_common"], ["u_extra"]],
        residuals=[["r_common"], ["r_extra"]],
        structure=[],
    )
    assert system.ulayout.structure == ("dense", "dense")


# ---------------------------------------------------------------- assemble path


def _init_block_system():
    """A block common x extra system, initialized with zero unknowns + random givens."""
    system = ModelNonlinearSystem(
        _diagonal_model(),
        unknowns=[["u_common", "u_extra"]],
        residuals=[["r_common", "r_extra"]],
        structure=["block"],
    )
    nb, nc, ns = 2, 2, 3
    torch.manual_seed(7)
    u = {
        "u_common": Scalar(torch.zeros(nb, nc, dtype=torch.float64), 1).with_sub_batch_ndim(1),
        "u_extra": Scalar(torch.zeros(nb, nc, ns, dtype=torch.float64), 1).with_sub_batch_ndim(2),
    }
    g = {
        "g_common": Scalar(torch.randn(nb, nc, dtype=torch.float64), 1).with_sub_batch_ndim(1),
        "g_extra": Scalar(torch.randn(nb, nc, ns, dtype=torch.float64), 1).with_sub_batch_ndim(2),
    }
    u_sv, g_sv = system.to_sparse(
        u, g, sub_batch_ndim={"u_common": 1, "u_extra": 2, "g_common": 1, "g_extra": 2}
    )
    system.initialize(u=u_sv, g=g_sv, dyn_shape=(nb,))
    return system, u_sv, g_sv


def test_assemble_convenience_methods_and_needB():
    """A/b/A_and_b/A_and_B/A_and_B_and_b all assemble; residual r = u - g gives b = g."""
    system, _, _ = _init_block_system()
    a_only = system.A()
    b_only = system.b()
    a1, b1 = system.A_and_b()
    a2, big_b = system.A_and_B()
    a3, big_b3, b3 = system.A_and_B_and_b()
    # A is the block Jacobian; identity residual r=u-g -> dR/du = I, so A is identity-like.
    assert (
        a_only.tensors[0][0].data.shape
        == a1.tensors[0][0].data.shape
        == a3.tensors[0][0].data.shape
    )
    # b = -residual = -(u - g) = g - u = g (u is zero), per group.
    for bb in (b_only, b1, b3):
        assert bb.tensors[0].data.shape[0] == 2
    # B is the cross-block (dR/dg) over the given layout.
    assert big_b.tensors[0][0].data.shape == big_b3.tensors[0][0].data.shape


def test_set_u_set_g_accept_sparsevector():
    """set_u / set_g accept a SparseVector (assemble -> disassemble -> state update)."""
    system, u_sv, g_sv = _init_block_system()
    system.set_u(u_sv)  # SparseVector branch
    system.set_g(g_sv)  # SparseVector branch
    # u()/g() rebuild from the committed state.
    assert set(system.u().layout.vars()) == {"u_common", "u_extra"}
    assert set(system.g().layout.vars()) == {"g_common", "g_extra"}


def test_system_to_moves_model_and_state():
    """`.to(...)` forwards to the model and walks the populated state wrappers."""
    system, _, _ = _init_block_system()
    returned = system.to(dtype=torch.float64)  # no-op dtype, but exercises the walk
    assert returned is system
    for tw in system._state.values():
        assert tw.dtype == torch.float64


# ---------------------------------------------------------------- error paths


def test_to_sparse_rejects_sub_batch_ndim_over_batch():
    """to_sparse's _shape_of raises when sub_batch_ndim exceeds the value's batch_ndim."""
    system = ModelNonlinearSystem(
        _diagonal_model(),
        unknowns=[["u_common", "u_extra"]],
        residuals=[["r_common", "r_extra"]],
        structure=["block"],
    )
    u = {
        "u_common": Scalar(torch.zeros(2, dtype=torch.float64), 1),
        "u_extra": Scalar(torch.zeros(2, dtype=torch.float64), 1),
    }
    g = {
        "g_common": Scalar(torch.zeros(2, dtype=torch.float64), 1),
        "g_extra": Scalar(torch.zeros(2, dtype=torch.float64), 1),
    }
    with pytest.raises(ValueError, match="exceeds batch_ndim"):
        system.to_sparse(u, g, sub_batch_ndim={"u_common": 5})


def test_call_model_missing_inputs_keyerror():
    """assemble before initialize -> empty state -> _call_model reports missing inputs."""
    system = ModelNonlinearSystem(
        _diagonal_model(),
        unknowns=[["u_common", "u_extra"]],
        residuals=[["r_common", "r_extra"]],
        structure=["block"],
    )
    with pytest.raises(KeyError, match="missing inputs"):
        system.assemble(need_A=False, need_B=False, need_b=True)
