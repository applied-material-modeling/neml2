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

r"""End-to-end correctness of the DENSE (common x extra) sub-batch fold.

The assembler keeps the outer block axis (common) diagonal and folds an inner,
DENSELY-coupled sub-batch axis (extra) into per-site base. No pre-existing scenario
exercises a dense extra axis -- the shipped common x extra scenarios only reach the
DIAGONAL partial-fold. These tests drive a block implicit solve whose per-extra-index
residual runs the unknown through :class:`DenseSubBatchMixing`
(``y_{g,i} = sum_r M_ir u_{g,r}``), so the per-common-site extra Jacobian block is a full
``extra x extra`` matrix. With ``M`` invertible the root is ``u_extra = M^{-1} g_extra``
per common, so Newton reaching it proves the dense block is assembled with the
correct extra-major K ordering: a transposed fold would corrupt the Newton step and
miss the root. ``M = I`` additionally pins that the dense code path (the model
always ``fullify``-s the extra axis, forcing ``k_state == 'full'``) reduces exactly
to the diagonal result.

Pure assembler mechanism (``DenseSubBatchMixing`` is physics-free), so this isolates
the fold from any specific model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# The dense-coupling triggers are test-only fixtures under tests/unit/_fold_fixtures.
sys.path.insert(0, str(Path(__file__).parent))
import importlib  # noqa: E402

importlib.import_module("_fold_fixtures")

from neml2 import load_string  # noqa: E402
from neml2.es.system import ModelNonlinearSystem  # noqa: E402
from neml2.types import Scalar  # noqa: E402


def _dense_matrix(n_extra: int, off: float = 0.3) -> torch.Tensor:
    """Invertible, symmetric dense mixing matrix ``I + off (J - I)``."""
    eye = torch.eye(n_extra, dtype=torch.float64)
    return eye + off * (1.0 - eye)


def _model_string(m: torch.Tensor, nb: int, n_common: int, n_extra: int):
    """Build the block common x extra scenario with a baked ``n_extra x n_extra`` matrix ``m``."""
    rows = ", ".join(repr(m[i].tolist()) for i in range(n_extra))
    return load_string(
        f"""
        [Settings]
          [example_batch_shape]
            u_common = '({nb}; {n_common})'
            g_common = '({nb}; {n_common})'
            u_extra  = '({nb}; {n_common}, {n_extra})'
            g_extra  = '({nb}; {n_common}, {n_extra})'
          []
        []
        [Tensors]
          [mmat]
            type = Python
            expr = 'Scalar(torch.tensor([{rows}], dtype=torch.float64)).sub_batch.retag(2)'
          []
        []
        [Models]
          [r_common]
            type = ScalarLinearCombination
            from = 'u_common g_common'
            to = 'r_common'
            weights = '1 -1'
          []
          [mix]
            type = DenseSubBatchMixing
            x = 'u_extra'
            mixing_matrix = 'mmat'
            y = 'mixed_extra'
          []
          [r_extra]
            type = ScalarLinearCombination
            from = 'mixed_extra g_extra'
            to = 'r_extra'
            weights = '1 -1'
          []
          [residual]
            type = ComposedModel
            models = 'r_common mix r_extra'
          []
        []
        [EquationSystems]
          [eq_sys]
            type = NonlinearSystem
            model = 'residual'
            unknowns  = 'u_common u_extra'
            residuals = 'r_common r_extra'
            structure = 'block'
          []
        []
        [Solvers]
          [lu]
            type = DenseLU
          []
          [newton]
            type = Newton
            linear_solver = 'lu'
            abs_tol = 1e-12
            rel_tol = 1e-10
            max_its = 25
          []
        []
        [Models]
          [model]
            type = ImplicitUpdate
            equation_system = 'eq_sys'
            solver = 'newton'
          []
        []
        """
    ).get_model("model")


def _solve(model, g_common: torch.Tensor, g_extra: torch.Tensor):
    """Run the block implicit solve; return (u_common, u_extra) as plain tensors."""
    nb, n_common = g_common.shape
    n_extra = g_extra.shape[-1]
    out = model.call_by_name(
        {
            "u_common": Scalar(
                torch.zeros(nb, n_common, dtype=torch.float64), 1
            ).with_sub_batch_ndim(1),
            "u_extra": Scalar(
                torch.zeros(nb, n_common, n_extra, dtype=torch.float64), 1
            ).with_sub_batch_ndim(2),
            "g_common": Scalar(g_common, 1).with_sub_batch_ndim(1),
            "g_extra": Scalar(g_extra, 1).with_sub_batch_ndim(2),
        }
    )
    return out["u_common"].data.reshape(nb, n_common), out["u_extra"].data.reshape(
        nb, n_common, n_extra
    )


def _closed_form_extra(m: torch.Tensor, g_extra: torch.Tensor) -> torch.Tensor:
    """Per-(batch, common) solution ``u = M^{-1} g_extra``."""
    nb, n_common, n_extra = g_extra.shape
    minv = torch.linalg.inv(m)
    return torch.stack(
        [torch.stack([minv @ g_extra[b, g] for g in range(n_common)]) for b in range(nb)]
    )


def test_dense_scenario_solves_to_closed_form():
    """A common x extra block solve with a dense M converges to M^{-1} g_extra per common-axis site.

    This is the operational pin on the dense-block K ordering: the per-common-site extra
    Jacobian is dense, so a transposed (extra <-> base) fold would break the Newton
    step and the solve would not reach the closed-form root.
    """
    torch.manual_seed(1)
    nb, n_common, n_extra = 2, 2, 3
    m = _dense_matrix(n_extra)
    g_common = torch.randn(nb, n_common, dtype=torch.float64)
    g_extra = torch.randn(nb, n_common, n_extra, dtype=torch.float64)
    model = _model_string(m, nb, n_common, n_extra)
    u_common, u_extra = _solve(model, g_common, g_extra)
    torch.testing.assert_close(u_common, g_common, rtol=0, atol=1e-9)
    torch.testing.assert_close(u_extra, _closed_form_extra(m, g_extra), rtol=0, atol=1e-9)


def test_identity_mixing_reduces_to_diagonal():
    """With M = I the dense path (the model always fullify-s the extra axis, forcing
    ``k_state == 'full'``) must reproduce the diagonal result ``u_extra = g_extra``."""
    torch.manual_seed(2)
    nb, n_common, n_extra = 2, 3, 4
    model = _model_string(torch.eye(n_extra, dtype=torch.float64), nb, n_common, n_extra)
    g_common = torch.randn(nb, n_common, dtype=torch.float64)
    g_extra = torch.randn(nb, n_common, n_extra, dtype=torch.float64)
    u_common, u_extra = _solve(model, g_common, g_extra)
    torch.testing.assert_close(u_common, g_common, rtol=0, atol=1e-9)
    torch.testing.assert_close(u_extra, g_extra, rtol=0, atol=1e-9)


@pytest.mark.parametrize("n_extra", [3, 5, 12])
def test_dense_fold_general_nextra(n_extra):
    """Nothing is hard-coded to a particular extra count: the dense fold solves the
    closed form for n_extra in {3, 5, 12}."""
    torch.manual_seed(100 + n_extra)
    nb, n_common = 2, 2
    m = _dense_matrix(n_extra, off=0.2)
    model = _model_string(m, nb, n_common, n_extra)
    g_common = torch.randn(nb, n_common, dtype=torch.float64)
    g_extra = torch.randn(nb, n_common, n_extra, dtype=torch.float64)
    _, u_extra = _solve(model, g_common, g_extra)
    torch.testing.assert_close(u_extra, _closed_form_extra(m, g_extra), rtol=0, atol=1e-9)


def _model_string_two_extra(m: torch.Tensor, nb: int, n_common: int, n_mid: int, n_extra: int):
    """Block scenario where the extra unknown carries TWO extra axes (p, extra), the
    dense mix acting on the inner (extra) one -- i.e. two folded axes of unequal
    extent (p != extra)."""
    rows = ", ".join(repr(m[i].tolist()) for i in range(n_extra))
    return load_string(
        f"""
        [Settings]
          [example_batch_shape]
            u_common = '({nb}; {n_common})'
            g_common = '({nb}; {n_common})'
            u_extra  = '({nb}; {n_common}, {n_mid}, {n_extra})'
            g_extra  = '({nb}; {n_common}, {n_mid}, {n_extra})'
          []
        []
        [Tensors]
          [mmat]
            type = Python
            expr = 'Scalar(torch.tensor([{rows}], dtype=torch.float64)).sub_batch.retag(2)'
          []
        []
        [Models]
          [r_common]
            type = ScalarLinearCombination
            from = 'u_common g_common'
            to = 'r_common'
            weights = '1 -1'
          []
          [mix]
            type = DenseSubBatchMixing
            x = 'u_extra'
            mixing_matrix = 'mmat'
            y = 'mixed_extra'
          []
          [r_extra]
            type = ScalarLinearCombination
            from = 'mixed_extra g_extra'
            to = 'r_extra'
            weights = '1 -1'
          []
          [residual]
            type = ComposedModel
            models = 'r_common mix r_extra'
          []
        []
        [EquationSystems]
          [eq_sys]
            type = NonlinearSystem
            model = 'residual'
            unknowns  = 'u_common u_extra'
            residuals = 'r_common r_extra'
            structure = 'block'
          []
        []
        [Solvers]
          [lu]
            type = DenseLU
          []
          [newton]
            type = Newton
            linear_solver = 'lu'
            abs_tol = 1e-12
            rel_tol = 1e-10
            max_its = 25
          []
        []
        [Models]
          [model]
            type = ImplicitUpdate
            equation_system = 'eq_sys'
            solver = 'newton'
          []
        []
        """
    ).get_model("model")


def test_dense_tangent_rejects_multi_extra_axis():
    """The dense tangent fold supports ONE folded extra axis; >1 raises a clear error.

    The vector fold is general in axis count (unit test), but the dense/row tangent
    K-fold's ordering is unverified for >1 extra axis -- a swap of two equal-length
    axes would be silent. Rather than risk that, the assembler rejects it explicitly.
    Here the extra unknown carries two extra axes (p=4, extra=3); assembling its dense
    Jacobian must raise NotImplementedError naming the single-extra-axis limit.
    """
    nb, n_common, n_mid, n_extra = 2, 2, 4, 3
    torch.manual_seed(5)
    model = _model_string_two_extra(_dense_matrix(n_extra), nb, n_common, n_mid, n_extra)
    # Non-zero givens so Newton assembles the Jacobian (zero residual would exit early).
    g_common = torch.randn(nb, n_common, dtype=torch.float64)
    g_extra = torch.randn(nb, n_common, n_mid, n_extra, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="one folded"):
        model.call_by_name(
            {
                "u_common": Scalar(
                    torch.zeros(nb, n_common, dtype=torch.float64), 1
                ).with_sub_batch_ndim(1),
                "u_extra": Scalar(
                    torch.zeros(nb, n_common, n_mid, n_extra, dtype=torch.float64), 1
                ).with_sub_batch_ndim(3),
                "g_common": Scalar(g_common, 1).with_sub_batch_ndim(1),
                "g_extra": Scalar(g_extra, 1).with_sub_batch_ndim(3),
            }
        )


def _model_string_diagonal(nb: int, n_common: int, n_extra: int):
    """Block common x extra scenario with a DIAGONAL extra coupling (``r_extra = u_extra -
    g_extra`` directly, no mixing), so the extra Jacobian block is per-index diagonal
    rather than dense -- the ``_convert_tangent_to_paired_block_bothextra`` (block-diagonal
    expand) path, as opposed to the dense ``_bothextra_dense`` path of ``_model_string``."""
    return load_string(
        f"""
        [Settings]
          [example_batch_shape]
            u_common = '({nb}; {n_common})'
            g_common = '({nb}; {n_common})'
            u_extra  = '({nb}; {n_common}, {n_extra})'
            g_extra  = '({nb}; {n_common}, {n_extra})'
          []
        []
        [Models]
          [r_common]
            type = ScalarLinearCombination
            from = 'u_common g_common'
            to = 'r_common'
            weights = '1 -1'
          []
          [r_extra]
            type = ScalarLinearCombination
            from = 'u_extra g_extra'
            to = 'r_extra'
            weights = '1 -1'
          []
          [residual]
            type = ComposedModel
            models = 'r_common r_extra'
          []
        []
        [EquationSystems]
          [eq_sys]
            type = NonlinearSystem
            model = 'residual'
            unknowns  = 'u_common u_extra'
            residuals = 'r_common r_extra'
            structure = 'block'
          []
        []
        [Solvers]
          [lu]
            type = DenseLU
          []
          [newton]
            type = Newton
            linear_solver = 'lu'
            abs_tol = 1e-12
            rel_tol = 1e-10
            max_its = 25
          []
        []
        [Models]
          [model]
            type = ImplicitUpdate
            equation_system = 'eq_sys'
            solver = 'newton'
          []
        []
        """
    ).get_model("model")


def test_diagonal_both_extra_block_solves_to_givens():
    """A diagonal (per-index) extra coupling drives the block-diagonal-expand bothextra
    path; the root is simply ``u = g`` on both the common and extra unknowns."""
    torch.manual_seed(3)
    nb, n_common, n_extra = 2, 3, 4
    model = _model_string_diagonal(nb, n_common, n_extra)
    g_common = torch.randn(nb, n_common, dtype=torch.float64)
    g_extra = torch.randn(nb, n_common, n_extra, dtype=torch.float64)
    u_common, u_extra = _solve(model, g_common, g_extra)
    torch.testing.assert_close(u_common, g_common, rtol=0, atol=1e-9)
    torch.testing.assert_close(u_extra, g_extra, rtol=0, atol=1e-9)


def _model_string_rowextra(nb: int, n_common: int, n_extra: int):
    """Block group {u_common:(common,), u_extra:(common, extra)} where the extra residual
    depends on the common-only unknown (``u_extra = g_extra - u_common`` per site). The
    ``(r_extra, u_common)`` Jacobian block is a block row WITH extra against a common-only
    column -- the ``_convert_tangent_to_paired_block_rowextra`` path."""
    return load_string(
        f"""
        [Settings]
          [example_batch_shape]
            u_common = '({nb}; {n_common})'
            g_common = '({nb}; {n_common})'
            u_extra  = '({nb}; {n_common}, {n_extra})'
            g_extra  = '({nb}; {n_common}, {n_extra})'
          []
        []
        [Models]
          [r_common]
            type = ScalarLinearCombination
            from = 'u_common g_common'
            to = 'r_common'
            weights = '1 -1'
          []
          [shift]
            type = CommonToExtraOffset
            e = 'u_extra'
            c = 'u_common'
            y = 'u_extra_shifted'
          []
          [r_extra]
            type = ScalarLinearCombination
            from = 'u_extra_shifted g_extra'
            to = 'r_extra'
            weights = '1 -1'
          []
          [residual]
            type = ComposedModel
            models = 'r_common shift r_extra'
          []
        []
        [EquationSystems]
          [eq_sys]
            type = NonlinearSystem
            model = 'residual'
            unknowns  = 'u_common u_extra'
            residuals = 'r_common r_extra'
            structure = 'block'
          []
        []
        [Solvers]
          [lu]
            type = DenseLU
          []
          [newton]
            type = Newton
            linear_solver = 'lu'
            abs_tol = 1e-12
            rel_tol = 1e-10
            max_its = 25
          []
        []
        [Models]
          [model]
            type = ImplicitUpdate
            equation_system = 'eq_sys'
            solver = 'newton'
          []
        []
        """
    ).get_model("model")


def test_row_extra_depends_on_common_solves_to_closed_form():
    """A per-(common, extra) residual coupled to the common-only unknown drives the
    row-extra tangent fold. Closed form: ``u_common = g_common`` and, from
    ``u_extra + u_common - g_extra = 0``, ``u_extra = g_extra - g_common`` per site."""
    torch.manual_seed(11)
    nb, n_common, n_extra = 2, 3, 4
    model = _model_string_rowextra(nb, n_common, n_extra)
    g_common = torch.randn(nb, n_common, dtype=torch.float64)
    g_extra = torch.randn(nb, n_common, n_extra, dtype=torch.float64)
    u_common, u_extra = _solve(model, g_common, g_extra)
    torch.testing.assert_close(u_common, g_common, rtol=0, atol=1e-9)
    expected_extra = g_extra - g_common.reshape(nb, n_common, 1)
    torch.testing.assert_close(u_extra, expected_extra, rtol=0, atol=1e-9)


def _block_extra_vs_dense_model():
    """Block group {u_common:(common,), u_extra:(common, extra)} plus a DENSE group
    holding the sub-batch-trivial scalar unknown ``s``. The extra residual depends on
    ``s`` (``r_extra = u_extra + s - g_extra``), so the ``(r_extra, s)`` Jacobian block
    is a block row WITH an extra axis against a DENSE column."""
    return load_string(
        """
        [Models]
          [r_common]
            type = ScalarLinearCombination
            from = 'u_common g_common'
            to = 'r_common'
            weights = '1 -1'
          []
          [add]
            type = AddScalarToExtra
            e = 'u_extra'
            s = 's'
            y = 'u_extra_shifted'
          []
          [r_extra]
            type = ScalarLinearCombination
            from = 'u_extra_shifted g_extra'
            to = 'r_extra'
            weights = '1 -1'
          []
          [r_s]
            type = ScalarLinearCombination
            from = 's g_s'
            to = 'r_s'
            weights = '1 -1'
          []
          [residual]
            type = ComposedModel
            models = 'r_common add r_extra r_s'
          []
        []
        """
    ).get_model("residual")


def test_block_extra_vs_dense_col_assembles():
    """A block-extra residual coupled to a sub-batch-trivial unknown in a DENSE column
    group drives the block-row-with-extra-against-a-dense-column tangent fold
    (``_convert_tangent_to_block`` ``row_structure == 'block'`` extra-into-row-storage
    branch). The Jacobian ``A = dR/du`` must assemble across the block + dense groups, and
    its ``(r_extra, s)`` cross block must equal the constant ``dR_extra/ds = 1``."""
    torch.manual_seed(13)
    nb, n_common, n_extra = 2, 2, 3
    system = ModelNonlinearSystem(
        _block_extra_vs_dense_model(),
        unknowns=[["u_common", "u_extra"], ["s"]],
        residuals=[["r_common", "r_extra"], ["r_s"]],
        structure=["block", "dense"],
    )
    u = {
        "u_common": Scalar(torch.zeros(nb, n_common, dtype=torch.float64), 1).with_sub_batch_ndim(
            1
        ),
        "u_extra": Scalar(
            torch.zeros(nb, n_common, n_extra, dtype=torch.float64), 1
        ).with_sub_batch_ndim(2),
        "s": Scalar(torch.zeros(nb, dtype=torch.float64), 1),  # dense (sub-batch-trivial)
    }
    g = {
        "g_common": Scalar(torch.randn(nb, n_common, dtype=torch.float64), 1).with_sub_batch_ndim(
            1
        ),
        "g_extra": Scalar(
            torch.randn(nb, n_common, n_extra, dtype=torch.float64), 1
        ).with_sub_batch_ndim(2),
        "g_s": Scalar(torch.randn(nb, dtype=torch.float64), 1),
    }
    u_sv, g_sv = system.to_sparse(
        u,
        g,
        sub_batch_ndim={
            "u_common": 1,
            "u_extra": 2,
            "s": 0,
            "g_common": 1,
            "g_extra": 2,
            "g_s": 0,
        },
    )
    system.initialize(u=u_sv, g=g_sv, dyn_shape=(nb,))
    a_mat = system.A()
    # Two row groups (block + dense) assembled; the (r_extra, s) cross block took the
    # block-row-extra-vs-dense-col fold. r_extra = u_extra + s - g_extra -> dR_extra/ds = 1.
    assert len(a_mat.tensors) == 2
    dre_ds = a_mat.disassemble().cells["r_extra"]["s"].data
    torch.testing.assert_close(dre_ds, torch.ones_like(dre_ds), rtol=0, atol=1e-12)
