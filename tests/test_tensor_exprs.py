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

"""Tests for [Tensors] Python-expression support.

The ``type = Python`` handler in ``_NativeInputFile.get_tensor`` evaluates
an ``expr`` field in a namespace pre-populated with the full
``neml2.types`` public namespace, ``torch``, ``math``, and optionally
``numpy``.  Multi-line code blocks are also supported; they must assign the
output to ``result``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from neml2.factory import _NativeInputFile, load_input, load_model
from neml2.types import SR2, Scalar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def _factory(tmp_path: Path, tensors_block: str) -> _NativeInputFile:
    return load_input(
        _write(
            tmp_path / "t.i",
            f"{tensors_block}\n[Models]\n[]\n",
        )
    )


# ---------------------------------------------------------------------------
# Single-expression forms
# ---------------------------------------------------------------------------


def test_scalar_expression(tmp_path):
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 5))'
  []
[]
""",
    )
    result = f.get_tensor("t")
    assert isinstance(result, Scalar)
    assert result.data.shape == (5,)
    assert result.data[0].item() == pytest.approx(0.0)
    assert result.data[-1].item() == pytest.approx(1.0)


def test_sr2_expression(tmp_path):
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = 'SR2(torch.zeros(3, 6))'
  []
[]
""",
    )
    result = f.get_tensor("t")
    assert isinstance(result, SR2)
    assert result.data.shape == (3, 6)


def test_raw_torch_tensor_returns_unwrapped(tmp_path):
    """A bare torch expression is returned as-is; wrapping is the call site's job."""
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = 'torch.linspace(0.0, 1.0, 5)'
  []
[]
""",
    )
    result = f.get_tensor("t")
    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, Scalar)


def test_float_literal(tmp_path):
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = '3.14'
  []
[]
""",
    )
    result = f.get_tensor("t")
    assert isinstance(result, float)


def test_math_in_namespace(tmp_path):
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = 'Scalar(torch.tensor(math.pi))'
  []
[]
""",
    )
    result = f.get_tensor("t")
    assert isinstance(result, Scalar)
    assert result.data.item() == pytest.approx(math.pi)


def test_native_types_free_function(tmp_path):
    """neml2.types free functions (tr, dev, …) are in the namespace."""
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = 'tr(SR2(torch.eye(6)))'
  []
[]
""",
    )
    result = f.get_tensor("t")
    # tr(I) in Mandel = diagonal entries xx+yy+zz = 1+1+1 = 3 (but stored as Scalar)
    assert isinstance(result, Scalar)


def test_numpy_in_namespace(tmp_path):
    pytest.importorskip("numpy")
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = 'Scalar(torch.tensor(np.linspace(0.0, 1.0, 5), dtype=torch.float64))'
  []
[]
""",
    )
    result = f.get_tensor("t")
    assert isinstance(result, Scalar)
    assert result.data.shape == (5,)
    assert result.data[-1].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multi-line code block
# ---------------------------------------------------------------------------


def test_multiline_code_block(tmp_path):
    f = load_input(
        _write(
            tmp_path / "t.i",
            """
[Tensors]
  [t]
    type = Python
    expr = '
      n = 7
      vals = torch.linspace(0.0, 1.0, n)
      result = Scalar(vals)
    '
  []
[]

[Models]
[]
""",
        )
    )
    result = f.get_tensor("t")
    assert isinstance(result, Scalar)
    assert result.data.shape == (7,)


def test_multiline_missing_result_raises(tmp_path):
    f = load_input(
        _write(
            tmp_path / "t.i",
            """
[Tensors]
  [t]
    type = Python
    expr = '
      x = 1 + 1
    '
  []
[]

[Models]
[]
""",
        )
    )
    with pytest.raises(ValueError, match="must assign its output to 'result'"):
        f.get_tensor("t")


# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------


def test_cross_reference(tmp_path):
    """Second tensor reads the first by using its name as a plain variable.

    HIT forbids quotes inside quoted strings, so ``tensor("name")`` cannot be
    written in a HIT expr field.  Instead, unknown identifiers in the expression
    are resolved via ``_TensorNamespace.__missing__``, which calls
    ``factory.get_tensor``.  No string literal is needed.
    """
    f = load_input(
        _write(
            tmp_path / "t.i",
            """
[Tensors]
  [base]
    type = Python
    expr = 'Scalar(torch.tensor(5.0))'
  []
  [doubled]
    type = Python
    expr = 'Scalar(base.data * 2.0)'
  []
[]

[Models]
[]
""",
        )
    )
    result = f.get_tensor("doubled")
    assert isinstance(result, Scalar)
    assert result.data.item() == pytest.approx(10.0)


def test_cross_reference_cycle_raises(tmp_path):
    f = load_input(
        _write(
            tmp_path / "t.i",
            """
[Tensors]
  [a]
    type = Python
    expr = 'a'
  []
[]

[Models]
[]
""",
        )
    )
    with pytest.raises(RecursionError, match="cross-references itself"):
        f.get_tensor("a")


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_eval_error_propagates(tmp_path):
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = '1 / 0'
  []
[]
""",
    )
    with pytest.raises(ValueError, match="ZeroDivisionError"):
        f.get_tensor("t")


def test_old_type_format_raises_helpful_error(tmp_path):
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Scalar
    values = '1.0'
  []
[]
""",
    )
    with pytest.raises(ValueError, match="expected a registered tensor type"):
        f.get_tensor("t")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_caching(tmp_path):
    """Two get_tensor calls for the same name return the identical object."""
    f = _factory(
        tmp_path,
        """
[Tensors]
  [t]
    type = Python
    expr = 'Scalar(torch.tensor(1.0))'
  []
[]
""",
    )
    a = f.get_tensor("t")
    b = f.get_tensor("t")
    assert a is b


def test_missing_raises_key_error(tmp_path):
    f = _factory(
        tmp_path,
        """
[Tensors]
[]
""",
    )
    with pytest.raises(KeyError, match="Tensors/missing"):
        f.get_tensor("missing")


# ---------------------------------------------------------------------------
# Integration: declare_typed_parameter mode 2 with raw-tensor auto-wrap
# ---------------------------------------------------------------------------


def test_raw_tensor_autowrap_via_declare_typed_parameter(tmp_path):
    """A raw torch.Tensor from the expression is auto-wrapped to the parameter's type_cls."""
    path = _write(
        tmp_path / "t.i",
        """
[Tensors]
  [E_val]
    type = Python
    expr = 'torch.tensor(1e5)'
  []
[]

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = 'E_val 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
""",
    )
    m = load_model(path, "elasticity")
    assert {n for n, _ in m.named_parameters()} >= {"E"}
    assert m.E.data.item() == pytest.approx(1e5)


def test_prewrapped_scalar_via_declare_typed_parameter(tmp_path):
    """A Scalar-wrapped expression passes through declare_typed_parameter without re-wrapping."""
    path = _write(
        tmp_path / "t.i",
        """
[Tensors]
  [E_val]
    type = Python
    expr = 'Scalar(torch.tensor(2e5))'
  []
[]

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = 'E_val 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
""",
    )
    m = load_model(path, "elasticity")
    assert {n for n, _ in m.named_parameters()} >= {"E"}
    assert m.E.data.item() == pytest.approx(2e5)
