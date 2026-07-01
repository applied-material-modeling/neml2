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

"""Unit tests for the native (py-eager) ``Model.jvp`` / ``Model.jacobian``.

These are the typed forward-mode input-derivative surface ``neml2.load_model``
returns -- the py-eager analogue of the ``forward`` / ``jvp`` / ``jacobian``
methods every other route exposes. The guarantees pinned here:

* **parity** -- value-for-value agreement with the trusted raw ``_EagerModel``
  adapter (which is itself pinned against the AOTI routes), so the typed native
  surface and the compiled/embedded routes never diverge;
* **typed returns** (CLAUDE.md Rule 1) -- jvp outputs are wrappers of the output
  type; Jacobian blocks are dynamic-base ``neml2.types.Tensor`` carrying the
  correct ``(batch, sub_batch, base)`` region split;
* **sub-batch support** -- unlike the plain-batch-only eager bridge, the native
  surface handles a sub-batched model (crystal-plasticity ``ResolvedShear``),
  with the per-site axis kept in the sub-batch region of the Jacobian block;
* **the implicit (IFT) path** -- a Newton-solved ``ImplicitUpdate`` matches
  ``torch.autograd``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import neml2
from neml2 import CrystalGeometry, ResolvedShear, cubic_symmetry_operators
from neml2.eager import _EagerModel
from neml2.types import R2, SR2, MillerIndex, Scalar
from neml2.types import Tensor as DynTensor

_REPO = Path(__file__).resolve().parents[2]
_SINGLE = _REPO / "tests" / "aoti" / "forward_single" / "model.i"  # SR2 strain -> SR2 stress
_COMPOSED = _REPO / "tests" / "aoti" / "forward_composed" / "model.i"  # 2-leaf, multi-output
_IMPLICIT = _REPO / "tests" / "aoti" / "implicit_simple" / "model.i"  # ImplicitUpdate (Newton)


def _raw_inputs(m, b):
    return {
        n: torch.randn(b, *base, dtype=m.dtype, device=m.device)
        for n, base in zip(m.input_names, m.input_base_shapes, strict=True)
    }


# --- parity with the trusted raw _EagerModel adapter --------------------------


@pytest.mark.parametrize("inp", [_SINGLE, _COMPOSED], ids=["single", "composed"])
def test_value_jvp_jacobian_match_eager(inp):
    """The typed native surface agrees value-for-value with the raw eager adapter
    on value, jvp, and every Jacobian block."""
    torch.manual_seed(0)
    native = neml2.load_model(str(inp), "model")
    eager = _EagerModel(str(inp), "model")
    b = 4
    raw = _raw_inputs(eager, b)
    tang = _raw_inputs(eager, b)

    nout, njac = native.jacobian(raw)
    eout, ejac = eager.jacobian(raw)
    assert set(nout) == set(eout)
    for o in eout:
        assert torch.allclose(nout[o].data, eout[o])
        for i in ejac[o]:
            assert isinstance(njac[o][i], DynTensor)  # Rule 1: typed block
            assert tuple(njac[o][i].data.shape) == tuple(ejac[o][i].shape)
            assert torch.allclose(njac[o][i].data, ejac[o][i], atol=1e-12)

    _, njvp = native.jvp(raw, tang)
    _, ejvp = eager.jvp(raw, tang)
    for o in ejvp:
        # jvp output is a wrapper of the output's type (not a raw tensor).
        assert type(njvp[o]).__name__ == type(nout[o]).__name__
        assert torch.allclose(njvp[o].data, ejvp[o], atol=1e-12)


def test_jacobian_matches_autograd():
    """The (stress, strain) block matches torch.autograd's Jacobian."""
    native = neml2.load_model(str(_SINGLE), "model")
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=next(native.parameters()).dtype)

    _, jac = native.jacobian({"strain": x})
    block = jac["stress"]["strain"]
    assert block.data.shape == (4, 6, 6)
    assert tuple(block.base_shape) == (6, 6)  # d SR2 / d SR2

    def f(t):
        r = native(SR2(t))
        return (r if not isinstance(r, tuple) else r[0]).data.sum(0)  # (6,)

    ref = torch.autograd.functional.jacobian(f, x).permute(1, 0, 2)  # (4, 6, 6)
    assert torch.allclose(block.data, ref, atol=1e-10)


def test_jvp_equals_jacobian_times_tangent():
    native = neml2.load_model(str(_SINGLE), "model")
    torch.manual_seed(0)
    dt = next(native.parameters()).dtype
    x = torch.randn(3, 6, dtype=dt)
    v = torch.randn(3, 6, dtype=dt)

    _, jvp_out = native.jvp({"strain": x}, {"strain": v})
    _, jac = native.jacobian({"strain": x})
    block = jac["stress"]["strain"].data  # (3, 6, 6)
    assert jvp_out["stress"].data.shape == (3, 6)
    assert torch.allclose(jvp_out["stress"].data, torch.einsum("bij,bj->bi", block, v), atol=1e-12)


def test_jvp_missing_tangent_is_zero():
    native = neml2.load_model(str(_SINGLE), "model")
    x = torch.ones(2, 6, dtype=next(native.parameters()).dtype)
    _, jvp_out = native.jvp({"strain": x}, {})  # no tangent seeded
    assert isinstance(jvp_out["stress"], SR2)
    assert jvp_out["stress"].data.shape == (2, 6)
    assert torch.count_nonzero(jvp_out["stress"].data) == 0


def test_composed_multi_output_block_shapes():
    native = neml2.load_model(str(_COMPOSED), "model")
    eager = _EagerModel(str(_COMPOSED), "model")
    torch.manual_seed(0)
    b = 5
    raw = _raw_inputs(eager, b)

    outputs, jac = native.jacobian(raw)
    assert set(outputs) == set(eager.output_names)
    for o_name, o_base in zip(eager.output_names, eager.output_base_shapes, strict=True):
        for i_name, i_base in zip(eager.input_names, eager.input_base_shapes, strict=True):
            assert tuple(jac[o_name][i_name].data.shape) == (b, *o_base, *i_base)


# --- sub-batch (crystal plasticity): native handles what the eager bridge can't


@pytest.fixture
def fcc_resolved_shear():
    fcc = CrystalGeometry(
        sym_ops=cubic_symmetry_operators(),
        lattice_vectors=torch.eye(3),
        slip_directions=MillerIndex(torch.tensor([1.0, 1.0, 0.0])),
        slip_planes=MillerIndex(torch.tensor([1.0, 1.0, 1.0])),
    )
    return ResolvedShear(crystal_geometry=fcc)


def test_subbatch_jacobian_region_split_and_numerics(fcc_resolved_shear):
    """A sub-batched output keeps its per-slip axis in the Jacobian block's
    SUB-batch region (not folded into plain batch), and the values match
    autograd. This is the regression guard for the batch_shape-vs-
    dynamic_batch_shape root fix."""
    leaf = fcc_resolved_shear
    stress_name, orient_name = list(leaf.input_spec)
    R = R2(torch.eye(3))
    sigma = SR2(torch.randn(4, 6))  # plain batch 4
    nslip = leaf(SR2(torch.randn(6)), R).data.shape[0]

    _, jac = leaf.jacobian({stress_name: sigma, orient_name: R})
    block = jac["resolved_shears"][stress_name]
    # d(scalar-per-slip)/d(SR2): batch (4,), sub (nslip,), base (6,).
    assert block.batch_ndim == 1
    assert tuple(block.batch_shape) == (4,)
    assert block.sub_batch_ndim == 1
    assert tuple(block.sub_batch_shape) == (nslip,)
    assert tuple(block.base_shape) == (6,)

    # numerics vs autograd (single, unbatched, for a clean per-slip Jacobian).
    s0 = torch.randn(6)
    _, jac0 = leaf.jacobian({stress_name: SR2(s0), orient_name: R})
    sl = s0.detach().clone().requires_grad_(True)
    ref = torch.autograd.functional.jacobian(lambda s: leaf(SR2(s), R).data, sl)  # (nslip, 6)
    assert torch.allclose(jac0["resolved_shears"][stress_name].data, ref, atol=1e-5)


def test_subbatch_jvp_is_output_typed(fcc_resolved_shear):
    """jvp on a sub-batched output returns an output-typed wrapper that keeps the
    per-slip sub-batch axis."""
    leaf = fcc_resolved_shear
    stress_name, orient_name = list(leaf.input_spec)
    R = R2(torch.eye(3))
    sigma = SR2(torch.randn(6))
    nslip = leaf(sigma, R).data.shape[0]

    _, jvp_out = leaf.jvp({stress_name: sigma, orient_name: R}, {stress_name: SR2(torch.randn(6))})
    rss = jvp_out["resolved_shears"]
    assert isinstance(rss, Scalar)
    assert rss.sub_batch_ndim == 1
    assert tuple(rss.sub_batch_shape) == (nslip,)


# --- implicit (IFT) path ------------------------------------------------------


def test_implicit_jacobian_matches_autograd():
    """A Newton-solved ImplicitUpdate: the chain-rule Jacobian (built on the
    implicit-function-theorem adjoint) matches torch.autograd through the solve."""
    native = neml2.load_model(str(_IMPLICIT), "model").to(torch.float64)
    names = list(native.input_spec)
    b = 3
    torch.manual_seed(2)
    # Physically-sane backward-Euler inputs (positive time step) so the 1-D
    # Newton solve converges; random t/t_old would give a garbage step.
    raw = {
        "x": torch.zeros(b, dtype=torch.float64),
        "x~1": torch.rand(b, dtype=torch.float64),
        "t": torch.full((b,), 1.0, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
        "x_rate": torch.rand(b, dtype=torch.float64),
    }

    _, jac = native.jacobian(raw)

    def f(*vals):
        r = native(*(Scalar(v) for v in vals))
        return (r if not isinstance(r, tuple) else r[0]).data

    leaves = [raw[n].detach().clone().requires_grad_(True) for n in names]
    J = torch.autograd.functional.jacobian(f, tuple(leaves))
    for k, n in enumerate(names):
        assert torch.allclose(jac["x"][n].data, torch.diagonal(J[k]), atol=1e-8)
