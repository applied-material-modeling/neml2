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

"""Comprehensive coverage for the sub-batch chain-rule sparsity machinery.

Three layers:

1. **Metadata propagation** — ``ComposedModel`` correctly composes
   ``list_deriv`` flags through chains, diamonds, multi-input/-output
   leaves, and ``VariableRemapper`` boundaries.

2. **Numerical correctness with sub-batch** — every shape of leaf
   (diagonal, sub-batch-preserving dense, sub-batch-reducing dense,
   sub-batch-reshaping dense) and every composition pattern reproduces
   the dense Jacobian computed via ``torch.autograd.functional.jacobian``.

3. **Backwards compatibility** — pre-sub-batch leaves continue to
   produce empty ``list_deriv``, the composed model gets the same
   structurally-zero metadata, and existing chain-rule paths are
   unaffected.
"""

from __future__ import annotations

from typing import cast

import torch

from neml2.chain_rule import combine_sparsity
from neml2.model import Model
from neml2.models.common import ComposedModel
from neml2.types import Scalar, sub_batch_sum

# ============================================================================
# Synthetic leaves used throughout the tests
# ============================================================================


class _ScalarScale(Model):
    """``y = a * x`` — pure broadcast, diagonal in both x and a."""

    def __init__(self, in_name: str, out_name: str, a: float = 2.5) -> None:
        super().__init__()
        self._in_name = in_name
        self._out_name = out_name
        self.input_spec = {in_name: Scalar}
        self.output_spec = {out_name: Scalar}
        self.register_typed_buffer("_a", Scalar(torch.tensor(a, dtype=torch.float64)))

    def forward(self, *inputs, v=None):  # type: ignore[override]
        (x,) = inputs
        y = self._a * x
        if v is None:
            return y
        actions = {self._in_name: lambda V, c=self._a: c * V}
        return y, self.apply_chain_rule(v, self._out_name, actions, output=y)


class _ScalarReduce(Model):
    """``y = sum_L x[..., L]`` — collapses the trailing sub-batch axis.

    Dense edge: the output (no sub-batch) depends on every input site.
    """

    list_deriv = {("y_total", "x_per_site"): "dense"}

    def __init__(self, in_name: str = "x_per_site", out_name: str = "y_total") -> None:
        super().__init__()
        self._in_name = in_name
        self._out_name = out_name
        self.input_spec = {in_name: Scalar}
        self.output_spec = {out_name: Scalar}
        if (out_name, in_name) != ("y_total", "x_per_site"):
            self.list_deriv = {(out_name, in_name): "dense"}

    def forward(self, *inputs, v=None):  # type: ignore[override]
        (x,) = inputs
        # Reduce the LAST sub-batch axis (which is the trailing batch axis
        # under the convention that sub-batch sits just before base).
        y_data = x.data.sum(dim=-1)
        y = Scalar(y_data)
        if v is None:
            return y

        def action(V):
            return sub_batch_sum(V, -1)

        return y, self.apply_chain_rule(v, self._out_name, {self._in_name: action}, output=y)


class _ScalarShift(Model):
    """``y[L] = x[L] + x[(L+1) mod L_total]`` — sub-batch-preserving dense.

    Each output site couples to TWO input sites (left + right neighbour).
    Output sub-batch shape == input sub-batch shape.
    """

    list_deriv = {("y_local", "x_local"): "dense"}

    def __init__(self) -> None:
        super().__init__()
        self.input_spec = {"x_local": Scalar}
        self.output_spec = {"y_local": Scalar}

    @staticmethod
    def _shift_sum(data: torch.Tensor) -> torch.Tensor:
        shifted = torch.roll(data, shifts=-1, dims=-1)
        return data + shifted

    def forward(self, *inputs, v=None):  # type: ignore[override]
        (x,) = inputs
        y = Scalar(self._shift_sum(x.data))
        if v is None:
            return y

        def action(V):
            d = V.data
            return Scalar(d + torch.roll(d, shifts=-1, dims=-1), sub_batch_ndim=V.sub_batch_ndim)

        return y, self.apply_chain_rule(v, "y_local", {"x_local": action}, output=y)


class _MultiOutputSplit(Model):
    """``y_dense = sum_L x[L]``, ``y_diag = x`` — mixed list_deriv per output.

    Confirms a multi-output leaf can declare different sparsity per
    output without spillover into the diagonal pair.
    """

    list_deriv = {("y_dense", "x"): "dense"}  # (y_diag, x) absent ⇒ diagonal

    def __init__(self) -> None:
        super().__init__()
        self.input_spec = {"x": Scalar}
        self.output_spec = {"y_dense": Scalar, "y_diag": Scalar}

    def forward(self, *inputs, v=None):  # type: ignore[override]
        (x,) = inputs
        y_dense = Scalar(x.data.sum(dim=-1))
        y_diag = 2.0 * x
        if v is None:
            return y_dense, y_diag

        v_dense = self.apply_chain_rule(
            v, "y_dense", {"x": lambda V: sub_batch_sum(V, -1)}, output=y_dense
        )
        v_diag = self.apply_chain_rule(v, "y_diag", {"x": lambda V: 2.0 * V}, output=y_diag)
        # The composed forward unpacks (*outputs, v_out_combined); merge.
        v_out = {**v_dense, **v_diag}
        return y_dense, y_diag, v_out


class _TwoInputAdd(Model):
    """``z = α·a + β·b`` — two-input diagonal leaf used in diamond tests."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        a_name: str = "a",
        b_name: str = "b",
        z_name: str = "z",
    ) -> None:
        super().__init__()
        self._a_name = a_name
        self._b_name = b_name
        self._z_name = z_name
        self.input_spec = {a_name: Scalar, b_name: Scalar}
        self.output_spec = {z_name: Scalar}
        self.alpha = alpha
        self.beta = beta

    def forward(self, *inputs, v=None):  # type: ignore[override]
        a, b = inputs
        z = Scalar(self.alpha * a.data + self.beta * b.data)
        if v is None:
            return z
        actions = {
            self._a_name: lambda V, α=self.alpha: α * V,
            self._b_name: lambda V, β=self.beta: β * V,
        }
        return z, self.apply_chain_rule(v, self._z_name, actions)


# ============================================================================
# 1. Metadata propagation through ComposedModel
# ============================================================================


def test_combine_sparsity_helper():
    assert combine_sparsity("diagonal") == "diagonal"
    assert combine_sparsity("diagonal", "diagonal") == "diagonal"
    assert combine_sparsity("dense") == "dense"
    assert combine_sparsity("diagonal", "dense") == "dense"
    assert combine_sparsity("dense", "diagonal", "diagonal") == "dense"


def test_single_diagonal_leaf_composed_is_empty():
    """A leaf with empty list_deriv composes to an empty map (= all diagonal)."""
    c = ComposedModel([_ScalarScale("x", "y")])
    assert c.list_deriv == {("y", "x"): "diagonal"}


def test_single_dense_leaf_composed_preserves_dense():
    c = ComposedModel([_ScalarReduce()])
    assert c.list_deriv == {("y_total", "x_per_site"): "dense"}


def test_linear_chain_all_diagonal():
    """Two-leaf chain: x → mid → y, both diagonal ⇒ master is diagonal."""
    c = ComposedModel([_ScalarScale("x", "mid", a=3.0), _ScalarScale("mid", "y", a=4.0)])
    assert c.list_deriv == {("y", "x"): "diagonal"}


def test_linear_chain_dense_then_diagonal_is_dense():
    """Dense step early in the chain propagates dense to the master output."""
    c = ComposedModel([_ScalarReduce(), _ScalarScale("y_total", "y", a=4.0)])
    # y depends on x_per_site via _ScalarReduce (dense) then _ScalarScale
    # (diagonal). Dense is absorbing.
    assert c.list_deriv == {("y", "x_per_site"): "dense"}


def test_linear_chain_diagonal_then_dense_is_dense():
    """Reversed order — diagonal then dense — still composes to dense."""
    c = ComposedModel([_ScalarScale("x", "x_scaled", a=2.0), _ScalarReduce_rename("x_scaled", "y")])
    assert c.list_deriv == {("y", "x"): "dense"}


def _ScalarReduce_rename(in_name: str, out_name: str) -> Model:
    """Test helper — construct _ScalarReduce with arbitrary input/output names."""
    return _ScalarReduce(in_name=in_name, out_name=out_name)


def test_two_independent_chains_metadata_disjoint():
    """Two completely independent leaves ⇒ no cross-talk in metadata."""
    s1 = _ScalarScale("a1", "b1", a=1.5)
    s2 = _ScalarScale("a2", "b2", a=2.5)
    c = ComposedModel([s1, s2])
    assert c.list_deriv == {("b1", "a1"): "diagonal", ("b2", "a2"): "diagonal"}


def _wrap_join(input_renames: dict[str, str], output_renames: dict[str, str]) -> Model:
    """Build a two-input add with the given external names.

    ``input_renames`` is ``{external: inner}`` (kept for call-site compatibility);
    here we just take the external keys and pass them as the model's actual names.
    """
    a_name, b_name = next(iter(input_renames)), list(input_renames)[1]
    (z_name,) = output_renames.values()
    return _TwoInputAdd(a_name=a_name, b_name=b_name, z_name=z_name)


def test_diamond_one_diagonal_one_dense_is_dense():
    """Diamond: x → mid1 (diag) and x → mid2 (dense) → join (diag).

    At the join, the dense path wins for the master pair (y, x).
    """
    top = _ScalarScale("x", "mid1", a=3.0)
    bottom = _ScalarReduce_rename("x", "mid2")
    join = _wrap_join({"mid1": "a", "mid2": "b"}, {"z": "y"})
    c = ComposedModel([top, bottom, join])
    assert c.list_deriv == {("y", "x"): "dense"}


def test_diamond_both_diagonal_stays_diagonal():
    """All-diagonal diamond keeps the master pair diagonal."""
    top = _ScalarScale("x", "mid1", a=3.0)
    bottom = _ScalarScale("x", "mid2", a=4.0)
    join = _wrap_join({"mid1": "a", "mid2": "b"}, {"z": "y"})
    c = ComposedModel([top, bottom, join])
    assert c.list_deriv == {("y", "x"): "diagonal"}


def test_multi_output_leaf_preserves_per_output_flags():
    """A multi-output leaf with one dense pair leaves the other pair diagonal."""
    c = ComposedModel([_MultiOutputSplit()])
    assert c.list_deriv == {
        ("y_dense", "x"): "dense",
        ("y_diag", "x"): "diagonal",
    }


def test_intermediate_variable_does_not_leak_into_master():
    """Intermediate variables consumed downstream don't appear in master list_deriv."""
    c = ComposedModel([_ScalarScale("x", "mid", a=3.0), _ScalarScale("mid", "y", a=4.0)])
    # mid is not in master input/output — only (y, x) appears.
    assert set(c.list_deriv.keys()) == {("y", "x")}


def test_additional_outputs_surface_intermediate_metadata():
    """An intermediate elevated to additional_outputs gets its own list_deriv entry."""
    c = ComposedModel(
        [_ScalarReduce(), _ScalarScale("y_total", "y", a=4.0)],
        additional_outputs=["y_total"],
    )
    assert c.list_deriv == {
        ("y", "x_per_site"): "dense",
        ("y_total", "x_per_site"): "dense",
    }


def test_nested_composed_passes_metadata_through():
    """An inner ComposedModel as a child contributes its own list_deriv."""
    inner = ComposedModel([_ScalarReduce()])
    # inner is a single-leaf composed with one dense edge (y_total, x_per_site).
    outer = ComposedModel([inner, _ScalarScale("y_total", "y", a=4.0)])
    assert outer.list_deriv == {("y", "x_per_site"): "dense"}


def test_resolver_handles_diagonal_then_diagonal_through_dense_join():
    """A two-input join leaf with both edges declared dense overrides upstream diagonals."""

    class _DenseJoin(Model):
        list_deriv = {("z", "a"): "dense", ("z", "b"): "diagonal"}

        def __init__(self) -> None:
            super().__init__()
            self.input_spec = {"a": Scalar, "b": Scalar}
            self.output_spec = {"z": Scalar}

        def forward(self, *inputs, v=None):  # type: ignore[override]
            a, b = inputs
            z = Scalar(a.data.sum(-1, keepdim=True) * b.data)
            if v is None:
                return z
            actions = {
                "a": lambda V: sub_batch_sum(V, -1, keepdim=True),
                "b": lambda V: V,
            }
            return z, self.apply_chain_rule(v, "z", actions, output=z)

    c = ComposedModel(
        [
            _ScalarScale("x", "a", a=2.0),
            _ScalarScale("y", "b", a=3.0),
            _DenseJoin(),
        ]
    )
    # (z, x) goes through (a, x) diagonal then (z, a) dense ⇒ dense.
    # (z, y) goes through (b, y) diagonal then (z, b) diagonal ⇒ diagonal.
    assert c.list_deriv == {("z", "x"): "dense", ("z", "y"): "diagonal"}


# ============================================================================
# 2. Numerical correctness with sub-batch tangents
# ============================================================================


def _seed_one_hot(t: torch.Tensor, *, sub_batch_ndim: int = 0) -> dict[str, Scalar]:
    """Build a single-input leading-K identity seed dict: ``{leaf: Scalar(...)}``.

    For ``Scalar`` inputs, the seed shape is ``(N, *t.shape)`` where leading
    ``N`` enumerates one tangent direction per scalar element of ``t``.
    ``sub_batch_ndim`` declares how many of ``t``'s trailing axes are
    per-sub-batch-site.
    """
    N = t.numel()
    data = torch.eye(N, dtype=t.dtype, device=t.device).reshape(N, *t.shape)
    return {"_seed": Scalar(data, sub_batch_ndim=sub_batch_ndim)}


def test_single_diagonal_leaf_with_sub_batch_matches_autograd():
    """y = a*x with x of shape (B=2, L=3): chain rule reproduces autograd."""
    m = _ScalarScale("x", "y", a=2.5)
    x_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    # Seed: one tangent column per element of x.
    seed = _seed_one_hot(x_data, sub_batch_ndim=1)  # shape (K=6, 2, 3)
    v = {"x": seed}
    _, v_out = m(Scalar(x_data, sub_batch_ndim=1), v=v)
    # Reference Jacobian: dy/dx is the diagonal (B*L, B*L) matrix with 2.5
    # on the diagonal; the chain-rule output rearranges this to per-site
    # tangents.
    expected = 2.5 * seed["_seed"].data
    assert torch.allclose(v_out["y"]["_seed"].data, expected)


def test_single_dense_reduction_leaf_jvp_matches_autograd():
    """y = sum_L x[L] reduces shape (B=2, L=3) → (B=2,). Chain rule via dense action."""
    m = _ScalarReduce()
    x_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    seed = _seed_one_hot(x_data, sub_batch_ndim=1)["_seed"]  # (K=6, 2, 3)
    v = {"x_per_site": {"src": seed}}
    _, v_out = m(Scalar(x_data, sub_batch_ndim=1), v=v)
    # Expected: dy/dx is a (B*1) x (B*L) matrix where each row sums its B
    # elements. The chain-rule output is sum_L V_in[..., L, n_in, k].
    expected = seed.data.sum(dim=-1)  # (K=6, 2)
    assert torch.allclose(v_out["y_total"]["src"].data, expected)
    # Cross-check against autograd: dy_b/dx_{b,L} = 1 if same b else 0.
    jac = cast(torch.Tensor, torch.autograd.functional.jacobian(lambda t: t.sum(dim=-1), x_data))
    # jac shape: (2,) output × (2, 3) input → (2, 2, 3). Apply to the
    # one-hot seed to confirm: y_out[b, 0, idx] = jac[b, b_seed, L_seed]
    # where idx encodes (b_seed, L_seed).
    for b in range(2):
        for b_s in range(2):
            for L_s in range(3):
                idx = b_s * 3 + L_s
                assert torch.isclose(v_out["y_total"]["src"].data[idx, b], jac[b, b_s, L_s])


def test_single_dense_neighbor_leaf_jvp_matches_autograd():
    """y[L] = x[L] + x[(L+1) mod L] — sub-batch-preserving dense edge."""
    m = _ScalarShift()
    x_data = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)  # B=1, L=4
    seed = _seed_one_hot(x_data, sub_batch_ndim=1)["_seed"]  # (K=4, 1, 4)
    v = {"x_local": {"s": seed}}
    y_w, v_out = m(Scalar(x_data, sub_batch_ndim=1), v=v)
    # Reference: y_data[L] = x[L] + x[(L+1) mod 4]
    y_expected = m._shift_sum(x_data)
    assert torch.allclose(y_w.data, y_expected)
    # Chain rule: dy[L]/dx[L'] = δ_{L,L'} + δ_{L+1 mod 4, L'}
    jac = cast(torch.Tensor, torch.autograd.functional.jacobian(m._shift_sum, x_data))
    # jac shape: (1, 4) × (1, 4) → (1, 4, 1, 4). Reorder the seed so that
    # for each (b, L_seed) seed column k, v_out[b, L, 0, k] equals
    # sum over L' of jac[b, L, b, L'] * seed[b, L', 0, k]. Verify directly.
    for L_out in range(4):
        for k in range(4):  # k indexes the (L_seed) flattened to size 4
            expected = jac[0, L_out, 0, k]  # 1.0 if k == L_out or k == (L_out+1) % 4 else 0
            assert torch.isclose(v_out["y_local"]["s"].data[k, 0, L_out], expected), (
                f"mismatch at L_out={L_out}, k={k}: got "
                f"{v_out['y_local']['s'].data[k, 0, L_out].item()} vs {expected.item()}"
            )


def test_chain_diagonal_diagonal_with_sub_batch_matches_autograd():
    """Two-step diagonal chain on sub-batch tensors composes correctly."""
    c = ComposedModel([_ScalarScale("x", "mid", a=3.0), _ScalarScale("mid", "y", a=4.0)])
    x_data = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)  # B=1, L=3
    seed = _seed_one_hot(x_data, sub_batch_ndim=1)["_seed"]
    v = {"x": {"s": seed}}
    _, v_out = c(Scalar(x_data, sub_batch_ndim=1), v=v)
    # Master Jacobian = 4.0 * 3.0 * I.
    expected = 12.0 * seed.data
    assert torch.allclose(v_out["y"]["s"].data, expected)


def test_chain_diagonal_then_dense_with_sub_batch_matches_autograd():
    """Diagonal scale then dense reduction: composed master pair is dense."""
    inner = _ScalarReduce_rename("x_scaled", "y")
    c = ComposedModel([_ScalarScale("x", "x_scaled", a=2.0), inner])
    x_data = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)  # B=1, L=3
    seed = _seed_one_hot(x_data, sub_batch_ndim=1)["_seed"]
    v = {"x": {"s": seed}}
    _, v_out = c(Scalar(x_data, sub_batch_ndim=1), v=v)
    # y = sum_L (2.0 * x[L]). dy/dx[L] = 2.0 for all L.
    expected = 2.0 * seed.data.sum(dim=-1)  # (K=3, 1)
    assert torch.allclose(v_out["y"]["s"].data, expected)
    # And the composed metadata records dense.
    assert c.list_deriv == {("y", "x"): "dense"}


def test_chain_dense_then_diagonal_with_sub_batch_matches_autograd():
    """Dense reduction first, then diagonal scale on a no-sub-batch result."""
    c = ComposedModel([_ScalarReduce(), _ScalarScale("y_total", "y", a=5.0)])
    x_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    seed = _seed_one_hot(x_data, sub_batch_ndim=1)["_seed"]
    v = {"x_per_site": {"s": seed}}
    _, v_out = c(Scalar(x_data, sub_batch_ndim=1), v=v)
    # y[b] = 5.0 * sum_L x[b, L]. dy/dx is 5.0 within the same b, 0 across b.
    expected = 5.0 * seed.data.sum(dim=-1)  # (K=6, 2)
    assert torch.allclose(v_out["y"]["s"].data, expected)


def test_diamond_one_dense_path_jvp_matches_autograd():
    """Diamond with one diagonal path and one dense path through a join."""
    top = _ScalarScale("x", "mid1", a=3.0)
    bottom = _ScalarReduce_rename("x", "mid2")
    # Note: mid1 has sub-batch (L,), mid2 has none. Bare addition won't
    # broadcast — wrap bottom's output back to per-site via a tiling step.
    # For this test, use an explicit join that expands mid2 across L.

    class _ExpandJoin(Model):
        list_deriv = {("y", "mid1"): "diagonal", ("y", "mid2"): "dense"}

        def __init__(self, L: int) -> None:
            super().__init__()
            self.input_spec = {"mid1": Scalar, "mid2": Scalar}
            self.output_spec = {"y": Scalar}
            self._L = L

        def forward(self, *inputs, v=None):  # type: ignore[override]
            mid1, mid2 = inputs
            # mid2 has shape (B,); broadcast to (B, L).
            y_data = mid1.data + mid2.data.unsqueeze(-1).expand(*mid2.data.shape, self._L)
            y = Scalar(y_data, sub_batch_ndim=mid1.sub_batch_ndim)
            if v is None:
                return y

            def mid1_action(V):
                return V

            def mid2_action(V):
                return V.sub_batch.expand_at(self._L, -1)

            actions = {"mid1": mid1_action, "mid2": mid2_action}
            return y, self.apply_chain_rule(v, "y", actions, output=y)

    L = 3
    c = ComposedModel([top, bottom, _ExpandJoin(L)])
    assert c.list_deriv == {("y", "x"): "dense"}
    x_data = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)  # B=1, L=3
    seed = _seed_one_hot(x_data, sub_batch_ndim=1)["_seed"]
    v = {"x": {"s": seed}}
    _, v_out = c(Scalar(x_data, sub_batch_ndim=1), v=v)
    # y[b, L] = 3*x[b, L] + sum_{L'} x[b, L']
    # dy/dx[L, L'] = 3*δ_{L,L'} + 1
    expected = torch.zeros_like(v_out["y"]["s"].data)
    for L_out in range(L):
        for L_seed in range(L):
            expected[L_seed, 0, L_out] = 3.0 if L_out == L_seed else 0.0
            expected[L_seed, 0, L_out] += 1.0
    assert torch.allclose(v_out["y"]["s"].data, expected)


def test_multi_output_leaf_jvp_separates_per_output_paths():
    """Confirm the multi-output leaf computes both pairs correctly and
    they're carried separately through the composed v_out dict."""

    m = _MultiOutputSplit()
    x_data = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)  # B=1, L=3
    seed = _seed_one_hot(x_data, sub_batch_ndim=1)["_seed"]
    v = {"x": {"s": seed}}
    _, _, v_out = m(Scalar(x_data, sub_batch_ndim=1), v=v)
    # Dense pair: y_dense = sum_L x[L]; dy_dense/dx[L] = 1.
    expected_dense = seed.data.sum(dim=-1)  # (K=3, 1)
    assert torch.allclose(v_out["y_dense"]["s"].data, expected_dense)
    # Diagonal pair: y_diag = 2x; dy_diag/dx[L] = 2.
    expected_diag = 2.0 * seed.data
    assert torch.allclose(v_out["y_diag"]["s"].data, expected_diag)


# ============================================================================
# 3. Backwards compatibility
# ============================================================================


def test_pre_sub_batch_leaf_has_empty_list_deriv():
    """A vanilla leaf with no list_deriv attribute defined sees the empty default."""
    m = _ScalarScale("x", "y")
    assert m.list_deriv == {}


def test_composed_of_legacy_leaves_is_all_diagonal():
    """ComposedModel of pre-sub-batch leaves only contains diagonal entries."""
    c = ComposedModel([_ScalarScale("x", "mid", a=3.0), _ScalarScale("mid", "y", a=4.0)])
    assert all(flag == "diagonal" for flag in c.list_deriv.values())


def test_class_attribute_empty_list_deriv_does_not_share_state():
    """The class-level default ``{}`` is read-only — instances don't share it."""
    m1 = _ScalarScale("x", "y")
    m2 = _ScalarScale("p", "q")
    # Touching one instance's list_deriv shouldn't appear on the other.
    assert m1.list_deriv == {}
    assert m2.list_deriv == {}
    assert m1.list_deriv is m2.list_deriv  # class-level dict is shared by ref
    # but if a leaf opts in (declares its own dict at class level), instances
    # of that class get THEIR class's dict — confirm by example.
    assert _ScalarReduce.list_deriv == {("y_total", "x_per_site"): "dense"}
    assert _ScalarScale.list_deriv == {}
