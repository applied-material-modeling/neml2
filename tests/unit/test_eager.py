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
from neml2.cli.aoti_export import _var_infos
from neml2.eager import _EagerModel, _infer_device_dtype
from neml2.types import SR2
from neml2.types._boundary import broadcast_to_common_batch, check_tensor

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
# An interpolation leaf whose abscissa/ordinate grid parameters carry a knot
# sub-batch axis (declared via `.sub_batch.retag(1)`) -- exercises call-batch
# shape stripping of parameter sub-batch axes.
_INTERP = _REPO / "tests" / "models" / "common" / "ScalarLinearInterpolation.i"


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


def test_jacobian_interpolation_sub_batch_param():
    """An interpolation model's grid parameters (``abscissa``/``ordinate``) carry a
    knot *sub-batch* axis, NOT a call-batch axis. ``jacobian`` must strip that
    sub-batch axis when forming the common call batch; otherwise it broadcasts the
    knot count (here 100) against the input batch (8) and crashes. Regression for
    the eager interpolation shape error."""
    m = _EagerModel(str(_INTERP), "E")
    B = 8  # deliberately != the 100 interpolation knots
    T = torch.linspace(300.0, 1500.0, B, dtype=m.dtype, device=m.device)

    outputs, jac = m.jacobian({"T": T})
    block = jac["E"]["T"]
    assert block.shape == (B,)  # (*B, *out_base=(), *in_base=()); knot axis not leaked
    # The value half matches a plain forward.
    assert torch.allclose(outputs["E"], m.forward({"T": T})["E"])
    # dE/dT equals the (constant) slope of the piecewise-linear grid, per element.
    h = 1.0
    fd = (m.forward({"T": T + h})["E"] - m.forward({"T": T - h})["E"]) / (2 * h)
    assert torch.allclose(block.reshape(-1), fd.reshape(-1), atol=1e-6)


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


# --- parameter Jacobian d(output)/d(parameter) (reverse-mode AD) --------------
# The parameter surface is a SEPARATE path from the input chain rule: it
# differentiates outputs w.r.t. the model's calibration nn.Parameters via
# reverse-mode autograd (neml2.models.param_ad), addressed by named_parameters()
# qualified name. These pin it against central finite differences.


def _fd_param_block(m, inputs, o_name, p_name, h_rel=1e-6):
    """Central-difference d(o_name)/d(p_name) for a SCALAR parameter.

    Returns ``(*batch, *out_base)`` (scalar param contributes no trailing axis),
    matching ``param_jacobian``'s block convention. Perturbs the named parameter
    via ``functional_call`` so the model itself is never mutated.
    """
    from torch.func import functional_call

    typed = tuple(m.input_spec[n](inputs[n]) for n in m.input_names)
    p0 = dict(m._model.named_parameters())[p_name].detach()
    assert p0.numel() == 1, "FD helper handles scalar params only"
    v0 = float(p0.reshape(()))
    h = h_rel * max(abs(v0), 1.0)
    o_idx = m.output_names.index(o_name)

    def out_at(val):
        cp = {p_name: torch.as_tensor(val, dtype=m.dtype, device=m.device).reshape(p0.shape)}
        res = functional_call(m._model, cp, typed)
        res = res if isinstance(res, tuple) else (res,)
        return res[o_idx].data

    return (out_at(v0 + h) - out_at(v0 - h)) / (2 * h)


def _rel_err(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def test_named_parameters_lists_typed_params():
    m = _EagerModel(str(_INPUT), "model")
    np_ = m.named_parameters()
    # LinearIsotropicElasticity exposes E and nu as scalar typed parameters. The
    # unified parameter surface is the qualified-name -> base-shape map.
    assert set(np_) == set(m.parameter_base_shapes) == {"model.E", "model.nu"}
    assert m.parameter_base_shapes == {"model.E": [], "model.nu": []}
    for q in m.parameter_base_shapes:
        assert tuple(np_[q].shape) == ()  # scalar params, no batch


def test_native_model_parameter_base_shapes():
    """``Model.parameter_base_shapes`` -- the native (py-eager) read-only parameter
    surface -- is the qualified-name -> base-shape map (Scalars => []), keyed
    exactly like the model's own ``named_parameters()``. The native model reports
    its authored namespace (a bare leaf's params are ``E`` / ``nu``, matching torch
    ``named_parameters()``); the AOTI / cpp-eager deployment routes wrap a bare leaf
    and report ``model.E`` instead."""
    native = neml2.load_model(str(_INPUT), "model")
    pbs = native.parameter_base_shapes
    assert pbs == {"E": [], "nu": []}
    assert set(pbs) == set(dict(native.named_parameters()))


def test_native_model_set_parameter():
    """``Model.set_parameter`` copies into the live ``nn.Parameter`` (torch
    ``no_grad``), preserving identity / dtype / grad-tracking; the next forward
    reflects the new value. ``KeyError`` for an unregistered name."""
    native = neml2.load_model(str(_INPUT), "model")
    x = SR2(torch.ones(6, dtype=torch.float64))
    before = native(x).data.clone()
    p = dict(native.named_parameters())["E"]

    native.set_parameter("E", torch.tensor(200.0, dtype=torch.float64))
    assert float(dict(native.named_parameters())["E"].detach()) == 200.0
    # In-place copy: same Parameter object, still grad-tracking.
    assert dict(native.named_parameters())["E"] is p
    assert p.requires_grad
    # The value change is reflected in the next forward.
    assert not torch.allclose(before, native(x).data)

    with pytest.raises(KeyError):
        native.set_parameter("does_not_exist", torch.tensor(1.0, dtype=torch.float64))


def test_native_model_set_parameter_batch_shape_change():
    """``set_parameter`` supports promoting a scalar parameter to a per-batch
    ``(B,)`` value: ``copy_`` cannot resize, so the Parameter is replaced. The
    next forward then uses the per-batch values (a different value per element)."""
    native = neml2.load_model(str(_INPUT), "model")
    assert tuple(dict(native.named_parameters())["E"].shape) == ()  # scalar to start

    perbatch = torch.tensor([100.0, 150.0, 200.0, 250.0], dtype=torch.float64)
    native.set_parameter("E", perbatch)
    E = dict(native.named_parameters())["E"]
    assert tuple(E.shape) == (4,)
    assert torch.equal(E.detach(), perbatch)
    assert E.requires_grad  # still a grad-tracking Parameter after the replace

    # The per-batch E feeds a batched forward and varies the output across batch.
    out = native(SR2(torch.ones(4, 6, dtype=torch.float64))).data
    assert tuple(out.shape) == (4, 6)
    assert not torch.allclose(out[0], out[1])


def test_eager_model_set_parameter():
    """``_EagerModel.set_parameter`` forwards to the native model; the read-side
    ``named_parameters`` reflects the new value."""
    m = _EagerModel(str(_INPUT), "model")
    q = next(iter(m.parameter_base_shapes))
    m.set_parameter(q, torch.tensor(123.0, dtype=m.dtype, device=m.device))
    assert float(m.named_parameters()[q]) == 123.0


def test_param_jacobian_matches_fd():
    """d(stress)/d(E), d(stress)/d(nu) match central finite differences."""
    m = _EagerModel(str(_INPUT), "model")
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=m.dtype, device=m.device)

    outputs, P = m.param_jacobian({"strain": x})
    # Value half matches a plain forward.
    assert torch.allclose(outputs["stress"], m.forward({"strain": x})["stress"])
    for p in ("model.E", "model.nu"):
        block = P["stress"][p]
        assert tuple(block.shape) == (4, 6)  # (*B, *out_base, *param_base=())
        ref = _fd_param_block(m, {"strain": x}, "stress", p)
        assert _rel_err(block, ref) < 1e-6, f"d(stress)/d({p}) disagrees with FD"


def test_param_jacobian_composed():
    """Parameter Jacobian threads through a 2-leaf ComposedModel: a leaf param
    drives every downstream output."""
    m = _EagerModel(str(_COMPOSED), "model")
    torch.manual_seed(0)
    b = 5
    ins = _rand_inputs(m, b)

    outputs, P = m.param_jacobian(ins)
    assert set(P) == set(m.output_names)
    fwd = m.forward(ins)
    for o_name in m.output_names:
        assert torch.allclose(outputs[o_name], fwd[o_name])
    for o_name, o_base in zip(m.output_names, m.output_base_shapes, strict=True):
        for p in m.parameter_base_shapes:
            block = P[o_name][p]
            assert tuple(block.shape) == (b, *o_base)  # scalar params
            ref = _fd_param_block(m, ins, o_name, p)
            assert _rel_err(block, ref) < 1e-6, f"d({o_name})/d({p}) disagrees with FD"


def test_param_jacobian_batched_parameter():
    """A per-batch-element parameter (Scalar ``E`` set to ``(B,)``) is
    differentiated per element. The call batch is ``broadcast(input batches,
    parameter batches)``, so the value and every derivative are correctly shaped
    even when the parameter's batch exceeds the inputs' -- the case that used to
    emit the wrong batch shape / crash. Validated vs per-element finite differences
    (the discriminating test: a batch-summing bug passes a uniform-E FD but fails
    a per-element FD)."""
    B = 4
    # Two cases: input batched at B, and an UNBATCHED input whose output batches
    # solely on the parameter (the case that used to crash in jacobian/param_jacobian).
    for label, strain in (
        ("input (B,)", torch.randn(B, 6, dtype=torch.float64)),
        ("input ()", torch.randn(6, dtype=torch.float64)),
    ):
        m = _EagerModel(str(_INPUT), "model")
        E0 = torch.linspace(90.0, 110.0, B, dtype=m.dtype, device=m.device)
        m.set_parameter("model.E", E0.clone())

        out, P = m.param_jacobian({"strain": strain})
        assert tuple(out["stress"].shape) == (B, 6), label
        # The input Jacobian also batches on the parameter (used to size-mismatch).
        jblock = m.jacobian({"strain": strain})[1]["stress"]["strain"]
        assert tuple(jblock.shape) == (B, 6, 6), label
        block = P["stress"]["model.E"]
        assert tuple(block.shape) == (B, 6), label  # (*B, *out_base, *param_base=())

        # Per-element central differences: E_b only affects stress_b.
        h = 1e-6
        fd = torch.zeros(B, 6, dtype=m.dtype)
        for b in range(B):
            up = E0.clone()
            up[b] += h
            m.set_parameter("model.E", up)
            sp = m.forward({"strain": strain})["stress"]
            dn = E0.clone()
            dn[b] -= h
            m.set_parameter("model.E", dn)
            sm = m.forward({"strain": strain})["stress"]
            fd[b] = (sp[b] - sm[b]) / (2 * h)
        assert _rel_err(block, fd) < 1e-6, f"{label}: batched d(stress)/d(E) disagrees with FD"

        # param_vjp lands the adjoint at each parameter's natural shape: the batched
        # E gets a per-element gradient (B,); the scalar nu is summed over the batch.
        m.set_parameter("model.E", E0.clone())
        g = m.param_vjp({"strain": strain}, {"stress": torch.randn(B, 6, dtype=m.dtype)})
        assert tuple(g["model.E"].shape) == (B,), label
        assert tuple(g["model.nu"].shape) == (), label


# --- py-eager native Model param methods (the route-method surface) -----------
# ``neml2.load_model(...)`` returns a native ``Model``; these pin its
# ``param_jacobian`` / ``param_vjp`` methods -- the py-eager half of the
# cross-route method surface, delegating to ``models.param_ad``. The engine itself is
# FD-validated above via ``_EagerModel``; here we pin the native method plumbing
# (input wrapping, default param set, value return, vjp == <w, jacobian>).


def test_native_model_param_jacobian_and_vjp():
    from torch.func import functional_call

    native = neml2.load_model(str(_INPUT), "model")
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.float64)

    outputs, P = native.param_jacobian({"strain": x})
    # Value half == a plain native forward.
    assert torch.allclose(outputs["stress"], native(SR2(x)).data)
    # Default param set = every typed parameter (qualified names).
    params = [q for q, _ in native.named_parameters()]
    assert set(P["stress"]) == set(params)

    for p in params:
        p0 = dict(native.named_parameters())[p].detach()
        v0 = float(p0.reshape(()))
        h = 1e-6 * max(abs(v0), 1.0)

        def out_at(val, _p=p, _p0=p0):
            cp = {_p: torch.as_tensor(val, dtype=torch.float64).reshape(_p0.shape)}
            return functional_call(native, cp, (SR2(x),)).data

        fd = (out_at(v0 + h) - out_at(v0 - h)) / (2 * h)
        assert _rel_err(P["stress"][p].flatten(), fd.flatten()) < 1e-6, f"d(stress)/d({p}) vs FD"

    # param_vjp adjoint == <w, param_jacobian>.
    w = torch.randn(4, 6, dtype=torch.float64)
    g = native.param_vjp({"strain": x}, {"stress": w})
    assert set(g) == set(params)
    for p in params:
        contracted = float((w.flatten() * P["stress"][p].flatten()).sum())
        assert abs(float(g[p]) - contracted) < 1e-9, f"param_vjp[{p}] != <w, J>"


def test_native_model_param_methods_subset_selection():
    """The optional ``params=`` selector restricts the columns / keys."""
    native = neml2.load_model(str(_INPUT), "model")
    x = torch.randn(4, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    _, P = native.param_jacobian({"strain": x}, params=["E"])
    assert set(P["stress"]) == {"E"}
    g = native.param_vjp(
        {"strain": x}, {"stress": torch.ones(4, 6, dtype=torch.float64)}, params=["nu"]
    )
    assert set(g) == {"nu"}


# --- implicit-model Jacobian vs finite differences ----------------------------
# Regression for the leading-K (``k_ndim``) seed-region convention. The Jacobian
# seed declares its identity directions as a K region, and derivative-leaf models
# (e.g. ``Normality``) emit tangents carrying that region. When such a leaf
# output is frozen and passed as a *given* to an ``ImplicitUpdate`` solve (radial
# return), its tangent is summed -- in ``_output_sensitivities`` -- with the
# direct input tangent for the same seed. If the seed left K folded into the
# batch (``k_ndim=0``) while the leaf used ``k_ndim=1``, ``align_k`` could not
# reconcile the two and produced a malformed block. These composed models
# exercise that path through the ``d(output)/d(E)`` column (``E`` feeds both the
# frozen ``flow_direction`` and the implicit residual directly). Implicit-model
# Jacobians were previously covered only for forward values, so this was latent.

_SM_REGRESSION = _REPO / "tests" / "regression" / "solid_mechanics"
_IMPLICIT_JAC_MODELS = [
    _SM_REGRESSION / "viscoplasticity" / "radial_return" / "model.i",
    _SM_REGRESSION / "viscoplasticity" / "isoharden" / "model.i",
    _SM_REGRESSION / "viscoplasticity" / "chaboche" / "model.i",
    _SM_REGRESSION / "rate_independent_plasticity" / "perfect" / "model.i",
    _SM_REGRESSION / "rate_independent_plasticity" / "radial_return" / "model.i",
]


def _fd_strain_column(m, inputs, o_name, h=1e-7):
    """Central-difference ``d(o_name)/d(E)`` -> ``(*batch, *out_base, 6)``."""
    base = inputs["E"]
    cols = []
    for k in range(6):
        e = torch.zeros_like(base)
        e[..., k] = h
        fp = m.forward({**inputs, "E": base + e})[o_name]
        fm = m.forward({**inputs, "E": base - e})[o_name]
        cols.append((fp - fm) / (2 * h))
    return torch.stack(cols, dim=-1)


@pytest.mark.parametrize(
    "path", _IMPLICIT_JAC_MODELS, ids=lambda p: f"{p.parents[1].name}-{p.parent.name}"
)
def test_implicit_jacobian_strain_column_matches_fd(path):
    m = _EagerModel(str(path), "model")
    b = 4
    # Uniaxial strain well into the plastic regime (yield strain ~1e-3), with all
    # lagged states left at the fresh-start zero and t = 1 (dt = 1). Perturbing
    # only E avoids predictor-only inputs (e.g. gamma~1), whose converged-solution
    # sensitivity is a structural zero the central difference cannot resolve.
    inputs = {}
    for name, base_shape in zip(m.input_names, m.input_base_shapes, strict=True):
        if name == "t":
            inputs[name] = torch.ones(b, *base_shape, dtype=m.dtype, device=m.device)
        elif name == "E":
            strain = torch.zeros(b, *base_shape, dtype=m.dtype, device=m.device)
            strain[..., 0] = 5e-3
            strain[..., 1] = strain[..., 2] = -0.3 * 5e-3
            inputs[name] = strain
        else:
            inputs[name] = torch.zeros(b, *base_shape, dtype=m.dtype, device=m.device)

    _, jac = m.jacobian(inputs)
    for o_name, o_base in zip(m.output_names, m.output_base_shapes, strict=True):
        block = jac[o_name]["E"]
        assert tuple(block.shape) == (b, *o_base, 6)
        ref = _fd_strain_column(m, inputs, o_name)
        scale = ref.abs().max().item() + 1e-30
        assert (block - ref).abs().max().item() / scale < 1e-5, (
            f"{path.parent.name}: d({o_name})/dE disagrees with finite differences"
        )


def _uniaxial_plastic_inputs(m, b=4):
    """Uniaxial strain into the plastic regime, lagged states at fresh-start
    zero, t = 1 -- the converging setup shared by the implicit derivative tests."""
    inputs = {}
    for name, base_shape in zip(m.input_names, m.input_base_shapes, strict=True):
        if name == "t":
            inputs[name] = torch.ones(b, *base_shape, dtype=m.dtype, device=m.device)
        elif name == "E":
            strain = torch.zeros(b, *base_shape, dtype=m.dtype, device=m.device)
            strain[..., 0] = 5e-3
            strain[..., 1] = strain[..., 2] = -0.3 * 5e-3
            inputs[name] = strain
        else:
            inputs[name] = torch.zeros(b, *base_shape, dtype=m.dtype, device=m.device)
    return inputs


def test_param_jacobian_through_implicit_matches_fd():
    """d(output)/d(parameter) composes through a Newton solve via the IFT adjoint.

    Checks parameters that live INSIDE the implicit residual (yield stress ``sy``,
    flow-rate ``eta`` -- their sensitivity needs ``du/dθ = -A⁻¹ ∂r/∂θ``) and one
    feeding the solve from the trial state (Young's modulus ``E``). The parameter
    surface (reverse-mode autograd) traverses ``_ImplicitUpdateFn.backward``'s
    implicit-function-theorem adjoint; this pins it against finite differences.
    """
    path = _SM_REGRESSION / "viscoplasticity" / "radial_return" / "model.i"
    m = _EagerModel(str(path), "model")
    inputs = _uniaxial_plastic_inputs(m, b=4)

    outputs, P = m.param_jacobian(inputs)
    fwd = m.forward(inputs)
    for o_name in m.output_names:
        assert torch.allclose(outputs[o_name], fwd[o_name])

    checks = {
        "model0.return_map.surface.yield_surface.sy": "implicit residual",
        "model0.return_map.surface.flow_rate.eta": "implicit residual",
        "model0.trial_state.cauchy_stress.E": "trial state -> given",
    }
    for q, where in checks.items():
        block = P["stress"][q]
        assert tuple(block.shape) == (4, 6)  # scalar param -> (*B, *out_base)
        ref = _fd_param_block(m, inputs, "stress", q)
        assert _rel_err(block, ref) < 1e-5, f"d(stress)/d({q}) [{where}] disagrees with FD"


def test_param_vjp_through_implicit_matches_jacobian_and_fd():
    """dL/dθ for L = sum_o <w_o, out_o> via the one-pass reverse-mode adjoint.

    The eager ``param_vjp`` swaps each parameter at its NATURAL shape (no
    per-batch expansion) and sums the gradient over the batch -- the global-
    parameter adjoint -- composing through the Newton solve's IFT backward. Pin
    it against the param-Jacobian contraction <w, dθ> (exact) and finite
    differences on the scalar loss.
    """
    path = _SM_REGRESSION / "viscoplasticity" / "radial_return" / "model.i"
    m = _EagerModel(str(path), "model")
    inputs = _uniaxial_plastic_inputs(m, b=4)

    _, P = m.param_jacobian(inputs)
    torch.manual_seed(0)
    cot = {
        o: torch.randn(4, *ob, dtype=m.dtype, device=m.device)
        for o, ob in zip(m.output_names, m.output_base_shapes, strict=True)
    }
    grads = m.param_vjp(inputs, cot)

    p0 = {q: m.named_parameters()[q].clone() for q in m.parameter_base_shapes}

    def loss():
        # Finite-difference probe: a plain scalar value, no autograd needed.
        with torch.no_grad():
            out = m.forward(inputs)
            return float(sum((out[o] * cot[o]).sum() for o in m.output_names))

    checks = [
        "model0.return_map.surface.yield_surface.sy",
        "model0.return_map.surface.flow_rate.eta",
        "model0.trial_state.cauchy_stress.E",
    ]
    for q in checks:
        g = float(grads[q])  # scalar param
        # Exact: contraction of the param Jacobian with the cotangents.
        contracted = sum((cot[o] * P[o][q]).sum().item() for o in m.output_names)
        assert abs(g - contracted) < 1e-7 * max(abs(contracted), 1.0), (
            f"param_vjp d(L)/d({q}) != <w, dθ> (vjp={g}, contracted={contracted})"
        )
        # Finite differences on the scalar loss.
        v0 = float(p0[q])
        h = 1e-6 * max(abs(v0), 1.0)
        m.named_parameters()[q].fill_(v0 + h)
        lp = loss()
        m.named_parameters()[q].fill_(v0 - h)
        lm = loss()
        m.named_parameters()[q].copy_(p0[q])
        fd = (lp - lm) / (2 * h)
        assert abs(g - fd) < 1e-4 * max(abs(fd), 1.0), (
            f"param_vjp d(L)/d({q}) disagrees with FD (vjp={g}, fd={fd})"
        )


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
