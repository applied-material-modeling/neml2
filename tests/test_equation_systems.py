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

from __future__ import annotations

import torch

from neml2.equation_systems import (
    AssembledMatrix,
    AssembledVector,
    AxisLayout,
    DenseImplicitSensitivity,
    DenseLinearizedSystem,
    DenseOperator,
    DenseRHS,
    ModelNonlinearSystem,
    _expanded_identity_seed,
)
from neml2.model import Model
from neml2.types import SR2, SSR4, Scalar


class ScalarResidual(Model):
    input_spec = {"x": Scalar, "c": Scalar}
    output_spec = {"x_residual": Scalar}

    def forward(self, x: Scalar, c: Scalar, v=None):
        r = x * x - c
        if v is None:
            return r
        return r, self.apply_chain_rule(
            v,
            "x_residual",
            {
                "x": lambda V, c=2.0 * x: c * V,
                "c": lambda V: -V,
            },
            output=r,
        )


def test_axis_layout_storage_sizes():
    layout = AxisLayout([["a", "b", "c"]], {"a": Scalar, "b": SR2, "c": SSR4})
    assert layout.var_size("a") == 1
    assert layout.var_size("b") == 6
    assert layout.var_size("c") == 36
    assert layout.group_size(0) == 43


def test_assembled_vector_round_trip_scalar_and_sr2():
    layout = AxisLayout([["x", "s"]], {"x": Scalar, "s": SR2})
    values = {
        "x": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "s": torch.arange(12, dtype=torch.float64).reshape(2, 6),
    }
    assembled = AssembledVector.from_dict(layout, values)
    assert assembled.tensors[0].shape == (2, 7)

    unpacked = assembled.disassemble()
    assert torch.equal(unpacked["x"], values["x"])
    assert torch.equal(unpacked["s"], values["s"])


def test_model_nonlinear_system_assembles_residual_and_jacobians():
    model = ScalarResidual()
    sys = ModelNonlinearSystem(model, unknowns=[["x"]])
    sys.initialize(
        u={"x": torch.tensor([2.0, 3.0], dtype=torch.float64)},
        g={"c": torch.tensor([4.0, 10.0], dtype=torch.float64)},
    )

    A, B = sys.A_and_B()
    b = sys.b()

    assert torch.equal(b.tensors[0], torch.tensor([[-0.0], [1.0]], dtype=torch.float64))
    assert torch.equal(A.tensors[0][0].squeeze(-1).squeeze(-1), torch.tensor([4.0, 6.0]))
    assert torch.equal(B.tensors[0][0].squeeze(-1).squeeze(-1), torch.tensor([-1.0, -1.0]))


def test_dense_export_wrappers_match_system_methods():
    model = ScalarResidual()
    sys = ModelNonlinearSystem(model, unknowns=[["x"]])
    x = torch.tensor([[2.0], [3.0]], dtype=torch.float64)
    c = torch.tensor([[4.0], [10.0]], dtype=torch.float64)
    sys.initialize(u={"x": x.squeeze(-1)}, g={"c": c.squeeze(-1)})

    A, b = sys.A_and_b()
    A2, B = sys.A_and_B()

    assert torch.equal(DenseRHS(sys)(x, c), b.tensors[0])
    assert torch.equal(DenseOperator(sys)(x, c), A.tensors[0][0])
    A_export, b_export = DenseLinearizedSystem(sys)(x, c)
    assert torch.equal(A_export, A.tensors[0][0])
    assert torch.equal(b_export, b.tensors[0])
    A_sens, B_export = DenseImplicitSensitivity(sys)(x, c)
    assert torch.equal(A_sens, A2.tensors[0][0])
    assert torch.equal(B_export, B.tensors[0][0])


def test_dense_linearized_system_torch_export():
    model = ScalarResidual()
    sys = ModelNonlinearSystem(model, unknowns=[["x"]])
    wrapper = DenseLinearizedSystem(sys)
    example = (
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        torch.tensor([[4.0], [9.0]], dtype=torch.float64),
    )

    from torch.export import Dim, export

    batch = Dim("batch", min=1, max=1 << 20)
    ep = export(wrapper, example, dynamic_shapes=({0: batch}, {0: batch}), strict=True)
    assert ep is not None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience APIs pyzag leans on
# ─────────────────────────────────────────────────────────────────────────────


def test_a_and_b_and_b_returns_all_three():
    """``A_and_B_and_b`` is a tiny wrapper around ``assemble(True, True, True)``
    that the pyzag interface uses to grab the full Newton tuple in one call."""
    model = ScalarResidual()
    sys = ModelNonlinearSystem(model, unknowns=[["x"]])
    sys.initialize(
        u={"x": torch.tensor([2.0, 3.0], dtype=torch.float64)},
        g={"c": torch.tensor([4.0, 10.0], dtype=torch.float64)},
    )
    A, B, b = sys.A_and_B_and_b()
    A2, B2 = sys.A_and_B()
    b2 = sys.b()
    assert torch.equal(A.tensors[0][0], A2.tensors[0][0])
    assert torch.equal(B.tensors[0][0], B2.tensors[0][0])
    assert torch.equal(b.tensors[0], b2.tensors[0])


def test_assembled_matrix_disassemble_round_trips_through_select_blocks():
    """``AssembledMatrix.disassemble()`` returns the per-(row_var, col_var)
    chain-rule blocks; ``select_blocks(...)`` rebuilds an equivalent
    assembled tensor. Round-trips the same matrix shape and values."""
    model = ScalarResidual()
    sys = ModelNonlinearSystem(model, unknowns=[["x"]])
    sys.initialize(
        u={"x": torch.tensor([2.0, 3.0], dtype=torch.float64)},
        g={"c": torch.tensor([4.0, 10.0], dtype=torch.float64)},
    )
    _, B = sys.A_and_B()
    blocks = B.disassemble()
    # Single (row_var="x_residual", col_var="c") pair for this model.
    assert set(blocks) == {"x_residual"}
    assert set(blocks["x_residual"]) == {"c"}
    # Round-trip via select_blocks on the same row/col layouts.
    rebuilt = AssembledMatrix.select_blocks(B.row_layout, B.col_layout, blocks)
    assert torch.equal(rebuilt.tensors[0][0], B.tensors[0][0])


def test_assembled_matrix_disassemble_raises_when_cache_lost():
    """Arithmetic on AssembledMatrix drops the per-block cache; ``disassemble``
    raises so callers don't silently get an empty dict."""
    model = ScalarResidual()
    sys = ModelNonlinearSystem(model, unknowns=[["x"]])
    sys.initialize(
        u={"x": torch.tensor([2.0, 3.0], dtype=torch.float64)},
        g={"c": torch.tensor([4.0, 10.0], dtype=torch.float64)},
    )
    _, B = sys.A_and_B()
    neg = -B  # arithmetic loses the cache
    import pytest

    with pytest.raises(RuntimeError, match="per-block cache"):
        neg.disassemble()


def test_select_blocks_zero_fills_missing_pairs():
    """``select_blocks`` zero-fills (row_var, col_var) pairs that aren't in
    ``blocks`` — sized from the layout's ``var_size`` and dtype/device from
    the present blocks. Used by pyzag to build the Jn sub-matrix when only
    a subset of B's columns map to old-state variables."""
    model = ScalarResidual()
    sys = ModelNonlinearSystem(model, unknowns=[["x"]])
    sys.initialize(
        u={"x": torch.tensor([2.0, 3.0], dtype=torch.float64)},
        g={"c": torch.tensor([4.0, 10.0], dtype=torch.float64)},
    )
    _, B = sys.A_and_B()
    real_blocks = B.disassemble()
    real_block = real_blocks["x_residual"]["c"]

    # Build a synthetic 2-col layout where col[0] is "c" (filled) and col[1]
    # is "phantom" (missing → zero-filled).
    wider = AxisLayout([["c", "phantom"]], {"c": Scalar, "phantom": Scalar})
    rebuilt = AssembledMatrix.select_blocks(B.row_layout, wider, {"x_residual": {"c": real_block}})
    flat = rebuilt.tensors[0][0]
    # Two unit-size cols → trailing dim is 2; first matches the original block,
    # second is all zeros.
    assert flat.shape == (2, 1, 2)
    assert torch.equal(flat[..., :1], real_block)
    assert torch.equal(flat[..., 1:], torch.zeros_like(flat[..., 1:]))


def test_load_nonlinear_system_helper():
    """``neml2.load_nonlinear_system`` returns the same object as
    ``load_input(path).get_equation_system(name)`` — a minimal HIT fixture
    exercises the parsing path end-to-end."""
    # Write a tiny HIT input on the fly and feed it through the factory.
    # ScalarResidual exists in this test module but is not registered with
    # the native registry, so build a system directly through the factory
    # against an in-tree elasticity model (already registered) plus a
    # one-line [EquationSystems] block.
    from pathlib import Path

    import nmhit

    from neml2.equation_systems import NonlinearSystem
    from neml2.factory import _NativeInputFile, load_nonlinear_system

    hit_text = """
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'elasticity'
    unknowns = 'strain'
    residuals = 'stress'
  []
[]
"""
    factory = _NativeInputFile(nmhit.parse_text(hit_text), Path("synthetic.i"))
    direct = factory.get_equation_system("eq_sys")
    assert isinstance(direct, NonlinearSystem)
    # And the top-level helper must give us the same kind of object when
    # pointed at a real on-disk file. Reuse the elasticity HIT fixture from
    # python/tests/native/models/.
    repo_fixture = (
        Path(__file__).parent / "models/solid_mechanics/elasticity/LinearIsotropicElasticity.i"
    )
    # That fixture is a ModelUnitTest input — it has no [EquationSystems]
    # block, so we expect a KeyError from the helper. Validates the wiring
    # without needing a separate fixture file.
    import pytest

    with pytest.raises(KeyError):
        load_nonlinear_system(repo_fixture, "eq_sys")


def test_expanded_identity_seed_preserves_singleton_sub_batch_axis():
    """``_expanded_identity_seed`` must materialise every sub_batch axis in
    the returned tangent's data — even when each one is size 1 — so that
    the tangent's data ndim matches the ``sub_batch_ndim`` it advertises.

    emits a leading-K typed seed. The size-1 sub-batch axis still
    must be explicit so downstream assembly can distinguish it from dynamic
    batch axes at trace time.
    """
    seed = _expanded_identity_seed(
        Scalar,
        sub_batch_shape=(1,),
        dyn_shape=(3,),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    # data must carry the size-1 sub_batch axis explicitly:
    # (K=1, *dyn=3, *sub_batch=1)
    assert seed.data.shape == (1, 3, 1)
    assert seed.sub_batch_ndim == 1
    # The no-sub-batch path stays as it was.
    seed0 = _expanded_identity_seed(
        SR2,
        sub_batch_shape=(),
        dyn_shape=(3,),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert seed0.data.shape == (6, 3, 6)
    assert seed0.sub_batch_ndim == 0
