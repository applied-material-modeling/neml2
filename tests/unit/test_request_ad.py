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

"""Unit tests for ``request_AD`` -- the auto-derived first-order chain rule.

A value-only leaf that calls :meth:`~neml2.models.model.Model.request_AD` gets its
``forward(v=)`` chain rule supplied by reverse-mode autodiff (py-eager path:
:meth:`Model._ad_pushforward` -> :func:`neml2.models.input_ad.local_input_jacobians`
-> :func:`neml2.types._boundary.contract_jacobian_block`). These tests pin that the
result matches a hand-written analytic chain rule and closed-form Jacobians, that
it composes with analytic leaves, and that the documented limits raise clearly.
"""

from __future__ import annotations

import pytest
import torch

from neml2.cli.aoti_export import _leading_k_identity_seed
from neml2.factory import register_neml2_object
from neml2.models.chain_rule import ChainRuleDict
from neml2.models.common import ComposedModel
from neml2.models.model import Model
from neml2.schema import HitSchema, input, output
from neml2.types import SR2, Scalar, inner
from neml2.types._boundary import assemble_jacobian

DT = torch.float64
DEV = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Synthetic leaves (registered once at import)
# --------------------------------------------------------------------------- #
@register_neml2_object("_ADCube")
class _ADCube(Model):
    """Value-only nonlinear Scalar->Scalar leaf: y = x^3 (J = 3 x^2)."""

    hit = HitSchema(input("x", Scalar, "in"), output("y", Scalar, "out"))

    def __post_init__(self):
        self.request_AD()

    def forward(self, x, *promoted_params):  # type: ignore[override]
        return x * x * x


@register_neml2_object("_AnalyticCube")
class _AnalyticCube(Model):
    """Hand-written chain-rule twin of :class:`_ADCube` (the reference)."""

    hit = HitSchema(input("x", Scalar, "in"), output("y", Scalar, "out"))

    def forward(self, x, *promoted_params, v: ChainRuleDict | None = None):  # type: ignore[override]
        y = x * x * x
        if v is None:
            return y
        return y, self.apply_chain_rule(v, "y", {"x": lambda V: Scalar(3.0) * x * x * V}, output=y)


@register_neml2_object("_ADMix")
class _ADMix(Model):
    """Value-only multi-input leaf: out = a * (b*b), a:SR2, b:Scalar -> out:SR2.

    ``d(out)/d(a) = b^2 * I`` (a (6,6) block); ``d(out)/d(b) = 2 b * a`` (a (6,)
    block). Exercises multiple inputs of mixed type and a non-scalar output base.
    """

    hit = HitSchema(
        input("a", SR2, "in a"),
        input("b", Scalar, "in b"),
        output("out", SR2, "out"),
    )

    def __post_init__(self):
        self.request_AD()

    def forward(self, a, b, *promoted_params):  # type: ignore[override]
        return a * (b * b)


@register_neml2_object("_ADNormSq")
class _ADNormSq(Model):
    """Value-only SR2->Scalar leaf: y = inner(x, x) (J = 2 x)."""

    hit = HitSchema(input("x", SR2, "in"), output("y", Scalar, "out"))

    def __post_init__(self):
        self.request_AD()

    def forward(self, x, *promoted_params):  # type: ignore[override]
        return inner(x, x)


@register_neml2_object("_ADSubset")
class _ADSubset(Model):
    """Like :class:`_ADMix` but only requests ``d(out)/d(a)``."""

    hit = HitSchema(
        input("a", SR2, "in a"),
        input("b", Scalar, "in b"),
        output("out", SR2, "out"),
    )

    def __post_init__(self):
        self.request_AD(inputs=["a"])

    def forward(self, a, b, *promoted_params):  # type: ignore[override]
        return a * (b * b)


@register_neml2_object("_ScaleA")
class _ScaleA(Model):
    """Analytic linear leaf z = 2 x (upstream of an AD leaf in composition)."""

    hit = HitSchema(input("x", Scalar, "in"), output("z", Scalar, "out"))

    def forward(self, x, *promoted_params, v: ChainRuleDict | None = None):  # type: ignore[override]
        z = Scalar(2.0) * x
        if v is None:
            return z
        return z, self.apply_chain_rule(v, "z", {"x": lambda V: Scalar(2.0) * V}, output=z)


@register_neml2_object("_CubeB")
class _CubeB(Model):
    """Value-only AD leaf y = z^3 consuming an upstream ``z`` (composition)."""

    hit = HitSchema(input("z", Scalar, "in"), output("y", Scalar, "out"))

    def __post_init__(self):
        self.request_AD()

    def forward(self, z, *promoted_params):  # type: ignore[override]
        return z * z * z


# --------------------------------------------------------------------------- #
# Helper: dense per-(out,in) Jacobian via the native identity-seeded chain rule
# --------------------------------------------------------------------------- #
def _native_jacobian(model, args):
    in_names = list(model.input_spec)
    out_names = list(model.output_spec)
    batch = tuple(args[0].batch_shape)
    seed = {
        n: {n: _leading_k_identity_seed(t, batch, dtype=DT, device=DEV)}
        for n, t in zip(in_names, model.input_spec.values(), strict=True)
    }
    *outs, v_out = model(*args, v=seed)
    jac = assemble_jacobian(
        v_out, tuple(outs), out_names, model.output_spec, in_names, model.input_spec
    )
    return outs, jac


# --------------------------------------------------------------------------- #
# Correctness (closed-form references)
# --------------------------------------------------------------------------- #
def test_cube_jacobian_matches_analytic_and_closed_form():
    torch.manual_seed(0)
    x = Scalar(torch.rand(5, dtype=DT) + 0.5)
    (oad,), jad = _native_jacobian(_ADCube(), (Scalar(x.data),))
    (oan,), jan = _native_jacobian(_AnalyticCube(), (Scalar(x.data),))
    assert torch.allclose(oad.data, oan.data)
    assert torch.allclose(jad["y"]["x"], jan["y"]["x"])  # AD == hand-written
    assert torch.allclose(jad["y"]["x"], 3.0 * x.data**2)  # == 3 x^2


def test_mix_jacobian_multi_input_multi_base():
    torch.manual_seed(0)
    a = SR2(torch.randn(4, 6, dtype=DT))
    b = Scalar(torch.rand(4, dtype=DT) + 0.5)
    (out,), jac = _native_jacobian(_ADMix(), (SR2(a.data), Scalar(b.data)))
    assert torch.allclose(out.data, a.data * (b.data**2).unsqueeze(-1))
    eye = torch.eye(6, dtype=DT)
    assert torch.allclose(jac["out"]["a"], (b.data**2).reshape(4, 1, 1) * eye)  # b^2 I
    assert torch.allclose(jac["out"]["b"], 2.0 * b.data.unsqueeze(-1) * a.data)  # 2 b a


def test_normsq_jacobian():
    torch.manual_seed(0)
    x = SR2(torch.randn(3, 6, dtype=DT))
    (out,), jac = _native_jacobian(_ADNormSq(), (SR2(x.data),))
    assert torch.allclose(out.data, inner(x, x).data)
    assert torch.allclose(jac["y"]["x"], 2.0 * x.data)  # J = 2 x


def test_composition_ad_leaf_after_analytic_leaf():
    """AD leaf composed downstream of an analytic leaf: total dy/dx = 24 x^2."""
    torch.manual_seed(0)
    comp = ComposedModel([_ScaleA(), _CubeB()])
    assert list(comp.input_spec) == ["x"] and list(comp.output_spec) == ["y"]
    x = Scalar(torch.rand(4, dtype=DT) + 0.5)
    (y,), jac = _native_jacobian(comp, (Scalar(x.data),))
    assert torch.allclose(y.data, (2.0 * x.data) ** 3)
    assert torch.allclose(jac["y"]["x"], 24.0 * x.data**2)  # 3(2x)^2 * 2


def test_request_ad_subset_selection():
    """Only the requested (out,in) pair gets an AD block; the rest are zero."""
    m = _ADSubset()
    assert m._ad_pairs == {("out", "a")}
    a = SR2(torch.randn(2, 6, dtype=DT))
    b = Scalar(torch.rand(2, dtype=DT) + 0.5)
    (_out,), jac = _native_jacobian(m, (SR2(a.data), Scalar(b.data)))
    assert torch.allclose(jac["out"]["a"], (b.data**2).reshape(2, 1, 1) * torch.eye(6, dtype=DT))
    assert torch.count_nonzero(jac["out"]["b"]) == 0  # not requested -> structural zero


# --------------------------------------------------------------------------- #
# Single-direction (k_ndim=0) seeds -- the shape ModelUnitTest and the
# ``Vec(eye(n))`` idiom send. request_AD must accept exactly what a hand-written
# action accepts, so a request_AD leaf is indistinguishable at the ``v=`` boundary.
# --------------------------------------------------------------------------- #
def _native_jvp_single(model, args, tangents):
    """Single-direction (k_ndim=0) pushforward: one directional seed per input."""
    seed = {n: {n: t} for n, t in tangents.items()}
    *outs, v_out = model(*args, v=seed)
    return outs, v_out


def test_single_direction_jvp_matches_analytic_and_closed_form():
    """A k_ndim=0 tangent (one direction) contracts to J @ v -- matching the
    hand-written twin and the closed form 3 x^2 v, with no spurious K axis."""
    torch.manual_seed(0)
    x = Scalar(torch.rand(5, dtype=DT) + 0.5)
    v = Scalar(torch.rand(5, dtype=DT))
    _, vad = _native_jvp_single(_ADCube(), (Scalar(x.data),), {"x": Scalar(v.data)})
    _, van = _native_jvp_single(_AnalyticCube(), (Scalar(x.data),), {"x": Scalar(v.data)})
    assert vad["y"]["x"].k_ndim == 0  # single-direction in -> single-direction out
    assert torch.allclose(vad["y"]["x"].data, van["y"]["x"].data)  # AD == hand-written
    assert torch.allclose(vad["y"]["x"].data, 3.0 * x.data**2 * v.data)


def test_single_direction_multi_input_multi_base():
    """Single-direction tangents on a multi-input AD leaf contract per edge:
    d(out) from a is b^2 v_a; d(out) from b is (2 b a) v_b."""
    torch.manual_seed(0)
    a = SR2(torch.randn(4, 6, dtype=DT))
    b = Scalar(torch.rand(4, dtype=DT) + 0.5)
    va = SR2(torch.randn(4, 6, dtype=DT))
    vb = Scalar(torch.rand(4, dtype=DT))
    _, vout = _native_jvp_single(
        _ADMix(), (SR2(a.data), Scalar(b.data)), {"a": SR2(va.data), "b": Scalar(vb.data)}
    )
    assert vout["out"]["a"].k_ndim == 0
    assert torch.allclose(vout["out"]["a"].data, (b.data**2).unsqueeze(-1) * va.data)
    assert torch.allclose(vout["out"]["b"].data, (2.0 * b.data * vb.data).unsqueeze(-1) * a.data)


def test_model_unit_test_verifies_request_ad_leaf():
    """ModelUnitTest's directional JVP check (a single-direction seed cross-checked
    against torch.autograd) now passes for a request_AD leaf -- previously its
    k_ndim=0 tangent was rejected by the contraction."""
    from neml2.drivers.ModelUnitTest import ModelUnitTest

    m = _ADCube()
    (xname,) = tuple(m.input_spec)
    (yname,) = tuple(m.output_spec)
    x = Scalar(torch.rand(5, dtype=DT) + 0.5)
    report = ModelUnitTest(m, {xname: x}, expected_outputs={yname: Scalar(x.data**3)}).run()
    assert report.value_checks == 1
    assert report.jvp_checks == 1


# --------------------------------------------------------------------------- #
# Declaration validation + documented limits
# --------------------------------------------------------------------------- #
def test_request_ad_rejects_unknown_names():
    m = _ADCube()
    with pytest.raises(ValueError, match="not an output"):
        m.request_AD(outputs=["nope"])
    with pytest.raises(ValueError, match="not an input"):
        m.request_AD(inputs=["nope"])


def test_request_ad_rejects_second_order():
    x = Scalar(torch.rand(3, dtype=DT) + 0.5)
    seed = {"x": {"x": _leading_k_identity_seed(Scalar, (3,), dtype=DT, device=DEV)}}
    with pytest.raises(NotImplementedError, match="first-order only"):
        _ADCube()(Scalar(x.data), v=seed, v2={})


def test_request_ad_rejects_sub_batch():
    m = _ADNormSq()
    x = SR2(torch.randn(2, 3, 6, dtype=DT), sub_batch_ndim=1)  # sub-batch axis
    with pytest.raises(NotImplementedError, match="plain-batch only"):
        m(SR2(x.data, sub_batch_ndim=1), v={"x": {}})
