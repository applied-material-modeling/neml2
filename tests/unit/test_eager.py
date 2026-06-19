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

"""Unit tests for the eager (uncompiled) C++-facing adapter ``neml2.eager._EagerModel``.

These pin the Python half of the embedded-Python eager bridge independent of the
C++ ``neml2::eager::Model`` (which the ``tests/cpp/test_eager`` executable
covers). The key guarantee checked here is *parity*: the adapter reports the
same input/output names + flat sizes the AOTI metadata path (`_var_infos`) would
report for the same model, and its ``forward`` matches the native model's output
value-for-value.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import neml2
from neml2._eager_boundary import broadcast_to_common_batch, check_tensor
from neml2.cli.aoti_export import _var_infos
from neml2.eager import _EagerModel, _infer_device_dtype
from neml2.types import SR2

# tests/unit/test_eager.py -> repo root -> the forward-only fixture .i.
_INPUT = Path(__file__).resolve().parents[2] / "tests" / "aoti" / "forward_single" / "model.i"


def test_metadata_parity_with_var_infos():
    """Names/sizes come from the same helper the AOTI metadata uses, so an eager
    model and its compiled twin agree on the boundary contract."""
    m = _EagerModel(str(_INPUT), "model")

    native = neml2.load_model(str(_INPUT), "model")
    in_infos = _var_infos(native.input_spec)
    out_infos = _var_infos(native.output_spec)

    assert m.input_names == [i["name"] for i in in_infos] == ["strain"]
    assert m.output_names == [i["name"] for i in out_infos] == ["stress"]
    assert m.input_sizes == [i["var_size"] for i in in_infos] == [6]
    assert m.output_sizes == [i["var_size"] for i in out_infos] == [6]


def test_forward_matches_native_model():
    m = _EagerModel(str(_INPUT), "model")

    torch.manual_seed(0)
    strain = torch.randn(5, 6, dtype=m.dtype, device=m.device)
    out = m.forward({"strain": strain})

    assert set(out) == {"stress"}
    assert out["stress"].shape == (5, 6)
    assert out["stress"].dtype == m.dtype

    # Value-for-value parity with the directly-loaded native model.
    native = neml2.load_model(str(_INPUT), "model")
    ref = native(SR2(strain))
    ref_data = ref.data if hasattr(ref, "data") else ref[0].data
    assert torch.allclose(out["stress"], ref_data)


def test_forward_is_deterministic():
    m = _EagerModel(str(_INPUT), "model")
    strain = torch.ones(3, 6, dtype=m.dtype, device=m.device)
    first = m.forward({"strain": strain})["stress"]
    second = m.forward({"strain": strain})["stress"]
    assert torch.allclose(first, second)


def test_missing_input_raises():
    m = _EagerModel(str(_INPUT), "model")
    with pytest.raises(KeyError, match="missing input"):
        m.forward({})


def test_wrong_dtype_raises():
    m = _EagerModel(str(_INPUT), "model")
    strain = torch.ones(3, 6, dtype=torch.float32, device=m.device)
    with pytest.raises(TypeError, match="EagerModel"):
        m.forward({"strain": strain})


def test_unknown_model_name_raises():
    with pytest.raises(KeyError):
        _EagerModel(str(_INPUT), "does_not_exist")


def test_device_and_dtype_reported():
    m = _EagerModel(str(_INPUT), "model")
    # NEML2 builds typed parameters at float64 regardless of the torch default.
    assert m.dtype == torch.float64
    assert m.device.type == "cpu"


def test_device_override():
    # Exercises the device-override path (model.to + _infer_device_dtype override
    # branch). "cpu" keeps the test runnable on any machine.
    m = _EagerModel(str(_INPUT), "model", device="cpu")
    assert m.device.type == "cpu"
    strain = torch.zeros(2, 6, dtype=m.dtype, device=m.device)
    assert m.forward({"strain": strain})["stress"].device.type == "cpu"


# --- Direct unit tests of the shared boundary helpers + device/dtype inference.
# These cover the fallback/validation branches a single-input forward never
# reaches (device mismatch, no-hint message, batch broadcast, tensor-less or
# non-float models).


def test_infer_device_dtype_fallbacks():
    # First tensor is non-floating: device is taken from it, dtype falls back to
    # the torch default.
    m_int = nn.Module()
    m_int.register_buffer("ints", torch.zeros(3, dtype=torch.int64))
    dev, dt = _infer_device_dtype(m_int, None)
    assert dev.type == "cpu"
    assert dt == torch.get_default_dtype()

    # No tensors at all: both device and dtype fall back to the torch defaults.
    dev2, dt2 = _infer_device_dtype(nn.Module(), None)
    assert dev2 == torch.get_default_device()
    assert dt2 == torch.get_default_dtype()


def test_check_tensor_device_mismatch_without_hint():
    # A wrong-device tensor (meta vs cpu) hits the device branch; an empty hint
    # exercises the no-hint message path.
    bad = torch.zeros(6, device="meta")
    with pytest.raises(TypeError, match="device="):
        check_tensor(
            bad, "x", torch.device("cpu"), torch.float64, kind="input", context="T", hint=""
        )


def test_broadcast_expands_lower_rank_input():
    # 'a' is base-only (6,) and must broadcast up to the common (4,) dyn batch.
    raw = {"a": torch.zeros(6), "b": torch.zeros(4, 6)}
    out, common_dyn = broadcast_to_common_batch(raw, {"a": SR2, "b": SR2}, {"a": 0, "b": 0})
    assert tuple(common_dyn) == (4,)
    assert out["a"].shape == (4, 6)
    assert out["b"].shape == (4, 6)
