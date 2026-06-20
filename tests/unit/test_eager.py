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

_REPO = Path(__file__).resolve().parents[2]
# tests/unit/test_eager.py -> repo root -> the forward-only fixture .i.
_INPUT = _REPO / "tests" / "aoti" / "forward_single" / "model.i"
# A two-leaf ComposedModel (strain -> stress -> mandel_stress): multi-output.
_COMPOSED = _REPO / "tests" / "aoti" / "forward_composed" / "model.i"
# A minimal ImplicitUpdate (Newton solve) -- used to exercise the convergence path.
_IMPLICIT = _REPO / "tests" / "aoti" / "implicit_simple" / "model.i"
# A crystal-plasticity leaf whose CrystalGeometry introduces a per-slip sub-batch
# axis internally -- the eager runtime must reject it (plain-batch only).
_SUB_BATCH = (
    _REPO / "tests" / "models" / "solid_mechanics" / "crystal_plasticity" / "ResolvedShear.i"
)


def _rand_inputs(m, b):
    """Random plain-batch inputs at each variable's canonical (b, *base) shape."""
    return {
        n: torch.randn(b, *base, dtype=m.dtype, device=m.device)
        for n, base in zip(m.input_names, m.input_base_shapes, strict=True)
    }


def test_metadata_parity_with_var_infos():
    """Names/base-shapes come from the same helper the AOTI metadata uses, so an
    eager model and its compiled twin agree on the boundary contract."""
    m = _EagerModel(str(_INPUT), "model")

    native = neml2.load_model(str(_INPUT), "model")
    in_infos = _var_infos(native.input_spec)
    out_infos = _var_infos(native.output_spec)

    assert m.input_names == [i["name"] for i in in_infos] == ["strain"]
    assert m.output_names == [i["name"] for i in out_infos] == ["stress"]
    assert m.input_base_shapes == [i["base_shape"] for i in in_infos] == [[6]]
    assert m.output_base_shapes == [i["base_shape"] for i in out_infos] == [[6]]


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


# --- jvp / jacobian -----------------------------------------------------------


def test_jacobian_matches_autograd():
    """The unflattened variable-pair block matches torch.autograd's Jacobian."""
    m = _EagerModel(str(_INPUT), "model")
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=m.dtype, device=m.device)

    outputs, jac = m.jacobian({"strain": x})
    # Nested {out: {in: (*B, *out_base, *in_base)}}.
    block = jac["stress"]["strain"]
    assert block.shape == (4, 6, 6)
    # The value half matches a plain forward.
    assert torch.allclose(outputs["stress"], m.forward({"strain": x})["stress"])

    native = neml2.load_model(str(_INPUT), "model")

    def f(t):
        r = native(SR2(t))
        return (r.data if hasattr(r, "data") else r[0].data).sum(0)  # (6,)

    # d sum_b stress_k / d strain_{b,j} -> (6, 4, 6); permute to per-batch (4,6,6).
    ref = torch.autograd.functional.jacobian(f, x).permute(1, 0, 2)
    assert torch.allclose(block, ref, atol=1e-10)


def test_jvp_equals_jacobian_times_tangent():
    m = _EagerModel(str(_INPUT), "model")
    torch.manual_seed(0)
    x = torch.randn(3, 6, dtype=m.dtype, device=m.device)
    v = torch.randn(3, 6, dtype=m.dtype, device=m.device)

    _, jvp_out = m.jvp({"strain": x}, {"strain": v})
    _, jac = m.jacobian({"strain": x})

    # jvp output is base-shaped (*B, *out_base); compare to the block @ v.
    assert jvp_out["stress"].shape == (3, 6)
    block = jac["stress"]["strain"]  # (3, 6, 6)
    assert torch.allclose(jvp_out["stress"], torch.einsum("bij,bj->bi", block, v), atol=1e-12)


def test_jvp_missing_tangent_is_zero():
    """A missing tangent contributes nothing -> zero directional derivative."""
    m = _EagerModel(str(_INPUT), "model")
    x = torch.ones(2, 6, dtype=m.dtype, device=m.device)
    _, jvp_out = m.jvp({"strain": x}, {})  # no tangent seeded
    assert jvp_out["stress"].shape == (2, 6)
    assert torch.count_nonzero(jvp_out["stress"]) == 0


def test_jvp_jacobian_composed_multi_output():
    """jvp/jacobian on a 2-leaf ComposedModel: nested blocks + jvp==J@v."""
    m = _EagerModel(str(_COMPOSED), "model")
    torch.manual_seed(0)
    b = 5
    ins = _rand_inputs(m, b)
    tang = _rand_inputs(m, b)

    outputs, jac = m.jacobian(ins)
    assert set(outputs) == set(m.output_names)
    # Every (out, in) pair is an unflattened (b, *out_base, *in_base) block.
    for o_name, o_base in zip(m.output_names, m.output_base_shapes, strict=True):
        for i_name, i_base in zip(m.input_names, m.input_base_shapes, strict=True):
            assert tuple(jac[o_name][i_name].shape) == (b, *o_base, *i_base)

    # jvp == Σ_in block @ tangent, per output (all SR2 here: out_base=in_base=(6,)).
    _, jvp_out = m.jvp(ins, tang)
    for o_name in m.output_names:
        contribs = [
            torch.einsum("bij,bj->bi", jac[o_name][i_name], tang[i_name])
            for i_name in m.input_names
        ]
        expect = torch.stack(contribs, 0).sum(0)
        assert torch.allclose(jvp_out[o_name], expect, atol=1e-10)


# --- #2 plain-batch guard: reject sub-batch (crystal-plasticity) models -------


def test_sub_batch_model_rejected():
    m = _EagerModel(str(_SUB_BATCH), "model")
    with pytest.raises(NotImplementedError, match="sub_batch"):
        m.forward(_rand_inputs(m, 3))


# --- #3 convergence failure surfaces as the recoverable typed exception -------


def test_convergence_failure_raises_typed():
    from neml2.aoti import ConvergenceError  # registered RuntimeError subclass
    from neml2.models.common import ImplicitUpdate

    m = _EagerModel(str(_IMPLICIT), "model")
    # Force a guaranteed non-convergence: zero Newton iterations against the
    # (random, hence non-zero) initial residual.
    iu = next(s for s in m._model.modules() if isinstance(s, ImplicitUpdate))
    iu.solver.miters = 0

    torch.manual_seed(0)
    with pytest.raises(ConvergenceError):
        m.forward(_rand_inputs(m, 2))
    # Backward-compatible: still a RuntimeError.
    assert issubclass(ConvergenceError, RuntimeError)
