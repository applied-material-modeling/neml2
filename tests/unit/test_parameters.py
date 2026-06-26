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

"""Tests for the four HIT parameter-resolution modes.

Mirrors the C++ ``ParameterStore::declare_parameter`` taxonomy:

1. Literal numeric value
2. ``[Tensors]`` cross-reference (batched static value)
3. ``[Models]`` output wiring (parameter promoted to an input variable,
   provider auto-included by the parent ComposedModel)
4. Bare variable specifier with no matching tensor or model (parameter
   promoted directly to an input)

The fixtures are deliberately tiny — they don't depend on
``ThermalEigenstrain`` / ``SR2LinearCombination`` so the suite stays decoupled
from the broader port-the-tutorial work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.factory import load_input, load_model
from neml2.models.common import ScalarLinearInterpolation
from neml2.types import Scalar

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


@pytest.fixture
def mode1_literal(tmp_path):
    return _write(
        tmp_path / "m1.i",
        """
[Models]
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
[]
""",
    )


@pytest.fixture
def mode2_tensor(tmp_path):
    return _write(
        tmp_path / "m2.i",
        """
[Tensors]
  [K_val]
    type = Python
    expr = 'Scalar(torch.tensor(750.0))'
  []
[]

[Models]
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 'K_val'
  []
[]
""",
    )


@pytest.fixture
def mode2_batched(tmp_path):
    return _write(
        tmp_path / "m2b.i",
        """
[Tensors]
  [K_val]
    type = Python
    expr = 'Scalar(torch.tensor([500.0, 1000.0, 1500.0, 2000.0]))'
  []
[]

[Models]
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 'K_val'
  []
[]
""",
    )


@pytest.fixture
def mode3_model(tmp_path):
    return _write(
        tmp_path / "m3.i",
        """
[Tensors]
  [K_x]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 1000.0]))'
  []
  [K_y]
    type = Python
    expr = 'Scalar(torch.tensor([500.0, 1500.0]))'
  []
[]

[Models]
  [K_model]
    type = ScalarLinearInterpolation
    argument = 'temperature'
    abscissa = 'K_x'
    ordinate = 'K_y'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 'K_model'
  []
  [chain]
    type = ComposedModel
    models = 'isoharden'
  []
[]
""",
    )


@pytest.fixture
def mode4_input(tmp_path):
    return _write(
        tmp_path / "m4.i",
        """
[Models]
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 'K_input'
  []
  [chain]
    type = ComposedModel
    models = 'isoharden'
  []
[]
""",
    )


# ---------------------------------------------------------------------------
# Mode 1 — literal
# ---------------------------------------------------------------------------


def test_mode1_literal_registers_static_parameter(mode1_literal):
    m = load_model(mode1_literal, "isoharden")
    assert list(m.input_spec) == ["equivalent_plastic_strain"]
    assert {n for n, _ in m.named_parameters()} == {"K"}
    assert "K" not in m._promoted_params
    assert m.K.data.item() == pytest.approx(1000.0)

    eps = Scalar(torch.tensor(0.01, dtype=torch.float64))
    out = m(eps)
    assert out.data.item() == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Mode 2 — [Tensors] cross-reference
# ---------------------------------------------------------------------------


def test_mode2_tensor_xref_registers_static_parameter(mode2_tensor):
    m = load_model(mode2_tensor, "isoharden")
    assert list(m.input_spec) == ["equivalent_plastic_strain"]
    assert {n for n, _ in m.named_parameters()} == {"K"}
    assert m.K.data.item() == pytest.approx(750.0)


def test_mode2_tensor_xref_batched(mode2_batched):
    m = load_model(mode2_batched, "isoharden")
    assert m.K.data.shape == (4,)
    # Forward broadcasts: eps scalar against K (4,) -> output (4,).
    eps = Scalar(torch.tensor(0.01, dtype=torch.float64))
    out = m(eps)
    assert out.data.tolist() == pytest.approx([5.0, 10.0, 15.0, 20.0])


def test_get_tensor_caches(mode2_tensor):
    """Repeated factory.get_tensor calls return the same wrapper instance."""
    f = load_input(mode2_tensor)
    a = f.get_tensor("K_val")
    b = f.get_tensor("K_val")
    assert a is b


def test_get_tensor_missing(mode2_tensor):
    f = load_input(mode2_tensor)
    with pytest.raises(KeyError, match="Tensors/missing"):
        f.get_tensor("missing")


# ---------------------------------------------------------------------------
# Mode 3 — variable specifier referencing a [Models] entry
# ---------------------------------------------------------------------------


def test_mode3_model_output_promotes_to_input(mode3_model):
    m = load_model(mode3_model, "chain")
    # Outer composition exposes both `equivalent_plastic_strain` and
    # `temperature` (the K_model's argument) as inputs.
    assert set(m.input_spec) == {"equivalent_plastic_strain", "temperature"}
    # K is no longer a parameter of the host; the provider's parameters
    # (abscissa, ordinate) take its place.
    param_names = {n for n, _ in m.named_parameters()}
    assert any(n.endswith("abscissa") for n in param_names)
    assert any(n.endswith("ordinate") for n in param_names)
    assert not any(n.endswith(".K") or n == "K" for n in param_names)

    # K(T=500) = 1000 (midpoint of 500..1500); h = 1000 * 0.01 = 10.0
    inputs_dict = {
        "equivalent_plastic_strain": torch.tensor(0.01, dtype=torch.float64),
        "temperature": torch.tensor(500.0, dtype=torch.float64),
    }
    args = [inputs_dict[k] for k in m.input_spec]
    out = m(*args)
    out_tensor = out[0] if isinstance(out, tuple) else out
    # ComposedModel now returns typed wrappers; `.data` works on both
    # wrappers (returns the underlying torch.Tensor) and raw tensors.
    assert out_tensor.data.item() == pytest.approx(10.0)


def test_mode3_records_nl_param_metadata(mode3_model):
    """The host model's _promoted_params should point at the K_model provider."""
    f = load_input(mode3_model)
    host = f.get_model("isoharden")
    assert "K" in host._promoted_params
    pparam = host._promoted_params["K"]
    assert pparam.input_name == "K_model"  # provider's single output name
    assert isinstance(pparam.provider, ScalarLinearInterpolation)


def test_mode3_provider_auto_included(mode3_model):
    """ComposedModel.from_hit pulls the provider in even when not in `models=`."""
    f = load_input(mode3_model)
    chain = f.get_model("chain")
    # chain.models = 'isoharden' lists ONLY isoharden, but K_model must be
    # in the dependency graph (it provides `K_model` to isoharden).
    plan_names = [type(getattr(chain, attr)).__name__ for attr, *_ in chain._plan]
    assert "ScalarLinearInterpolation" in plan_names
    assert "LinearIsotropicHardening" in plan_names


# ---------------------------------------------------------------------------
# Mode 4 — bare variable specifier (no matching tensor or model)
# ---------------------------------------------------------------------------


def test_mode4_bare_variable_promotes_to_input(mode4_input):
    m = load_model(mode4_input, "chain")
    assert set(m.input_spec) == {"equivalent_plastic_strain", "K_input"}
    assert {n for n, _ in m.named_parameters()} == set()  # K is no longer a parameter

    inputs_dict = {
        "equivalent_plastic_strain": torch.tensor(0.01, dtype=torch.float64),
        "K_input": torch.tensor(1000.0, dtype=torch.float64),
    }
    args = [inputs_dict[k] for k in m.input_spec]
    out = m(*args)
    out_tensor = out[0] if isinstance(out, tuple) else out
    # ComposedModel now returns typed wrappers; `.data` works on both
    # wrappers (returns the underlying torch.Tensor) and raw tensors.
    assert out_tensor.data.item() == pytest.approx(10.0)


def test_mode4_records_nl_param_without_provider(mode4_input):
    f = load_input(mode4_input)
    host = f.get_model("isoharden")
    assert "K" in host._promoted_params
    pparam = host._promoted_params["K"]
    assert pparam.input_name == "K_input"
    assert pparam.provider is None  # mode 4 — no provider model


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unresolvable_string_without_factory_raises():
    """A pure string spec with no factory + no promotion-mode support should raise."""
    with pytest.raises(ValueError, match="cannot resolve"):
        # ScalarLinearInterpolation's parameters use allow_promotion=False;
        # without a factory there's no tensor or model lookup to fall back on.
        ScalarLinearInterpolation(abscissa="alpha_x", ordinate="alpha_y")
