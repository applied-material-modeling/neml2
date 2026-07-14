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

"""Compiled matrix-free sensitivity (IFT / ParamIFT) solves: metadata contract +
derivative correctness.

The derivative linear solves on an ``ImplicitUpdate`` are separately configurable
(``input_sensitivity_solver`` for ``du/dg``, ``param_sensitivity_solver`` for
``du/dθ``). When one is a matrix-free ``GMRES`` / ``BiCGStab``, the exporter omits
its baked ``_solve_ift`` / ``_solve_param`` graph and the C++ runtime instead runs
a Krylov solve over the assembled A (``krylov_solve_dense``). These tests pin (1)
that schema-v12 contract and (2) that the iterative derivative equals the direct
(DenseLU) derivative of the identical model -- correctness, not self-consistency.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.cli.aoti_export import _reverse_ad_aoti_unsupported_reason

# Parameter-derivative AOTI graphs (ParamIFT `dr_dparam`) lower reverse-mode
# autograd.grad, which needs torch >= 2.11 (`trace_autograd_ops`); skip the
# param-sensitivity cases on the exact predicate the exporter guards on so the
# test gate and the runtime guard cannot drift. The INPUT-sensitivity cases use
# the forward chain rule and run on every torch.
_PARAM_DERIV_UNSUPPORTED = _reverse_ad_aoti_unsupported_reason()
_REQUIRES_PARAM_DERIV_TORCH = pytest.mark.skipif(
    _PARAM_DERIV_UNSUPPORTED is not None,
    reason=f"reverse-mode AD AOTI compilation {_PARAM_DERIV_UNSUPPORTED}",
)

_HERE = Path(__file__).parent
# Same Perzyna physics; the *_deriv variant selects an iterative input-sensitivity
# solver, the plain one keeps the default (direct) sensitivity solve.
_GMRES = _HERE / "krylov_gmres" / "model.i"
_GMRES_DERIV = _HERE / "krylov_gmres_deriv" / "model.i"
# Scalar implicit param model; the *_gmres variant selects an iterative
# param-sensitivity solver, the plain one keeps the default direct solve.
_PARAM = _HERE / "implicit_param" / "model.i"
_PARAM_GMRES = _HERE / "implicit_param_gmres" / "model.i"
_QNAME = "residual.rate.weight_0"


def _implicit_seg(meta: dict) -> dict:
    return next(s for s in meta["segments"] if s.get("kind") == "implicit")


def _inputs_from_meta(meta: dict, b: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        v["name"]: torch.randn(
            b, *[int(s) for s in v.get("base_shape", [])], generator=gen, dtype=torch.float64
        )
        * 0.1
        for v in meta["inputs"]
    }


def test_eager_input_sensitivity_matches_direct():
    """py-eager: the iterative input_sensitivity_solver's Jacobian (GMRES `.solve`
    over the assembled A via the C++ bare-Krylov binding) equals the default direct
    DenseLU solve of the identical model. Fast (no compile)."""
    from neml2 import load_input

    def jac(path):
        m = load_input(str(path)).get_model("model")
        spec = m.input_spec
        gen = torch.Generator().manual_seed(0)
        raw = {
            n: torch.randn(4, *tuple(t.BASE_SHAPE), generator=gen) * 0.1 for n, t in spec.items()
        }
        return m.jacobian(raw)[1]

    jd = jac(_GMRES)  # default (direct) sensitivity solve
    jk = jac(_GMRES_DERIV)  # iterative GMRES input sensitivity
    for o in jd:
        for i in jd[o]:
            a = jk[o][i].data if hasattr(jk[o][i], "data") else jk[o][i]
            b = jd[o][i].data if hasattr(jd[o][i], "data") else jd[o][i]
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-8)


def test_eager_param_sensitivity_matches_direct():
    """py-eager: the iterative param_sensitivity_solver's parameter Jacobian (the
    reverse-mode adjoint routed through a Krylov Aᵀ solve) equals the default direct
    solve. Fast (no compile)."""
    from neml2.eager import _EagerModel

    b = 5
    ins = {
        "x": torch.zeros(b, dtype=torch.float64),
        "x~1": torch.linspace(1.0, 2.0, b, dtype=torch.float64),
        "t": torch.full((b,), 0.5, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
    }
    key = f"model.{_QNAME}"
    pd = _EagerModel(str(_PARAM), "model").param_jacobian(ins)[1]["x"][key]
    pk = _EagerModel(str(_PARAM_GMRES), "model").param_jacobian(ins)[1]["x"][key]
    assert torch.allclose(pk, pd, rtol=1e-7, atol=1e-9)


def test_input_sensitivity_krylov_metadata(tmp_path: Path):
    """An iterative input_sensitivity_solver emits `_jacobian_given` but NOT
    `_solve_ift`; the segment records input_sensitivity.kind == krylov and the
    per-pair flat offsets the C++ slice needs."""
    from neml2.cli.aoti_export import export_model_for_aoti

    meta = export_model_for_aoti(_GMRES_DERIV, "model", tmp_path / "art", derivatives=["stress:"])
    seg = _implicit_seg(meta)
    assert "jacobian_given_package" in seg
    assert "solve_ift_package" not in seg
    assert seg["input_sensitivity"]["kind"] == "krylov"
    assert seg["input_sensitivity"]["krylov"]["method"] == "gmres"
    for pair in seg["jacobian_pairs"]:
        assert "row_offset" in pair and "col_offset" in pair


def test_input_sensitivity_krylov_matches_direct(tmp_path: Path):
    """The compiled iterative-IFT input Jacobian equals the direct (DenseLU) IFT
    Jacobian of the identical Perzyna model -- correctness of the C++ Krylov solve
    over the assembled A."""
    from neml2.aoti import Model
    from neml2.cli.aoti_export import export_model_for_aoti

    direct_out = tmp_path / "direct"
    krylov_out = tmp_path / "krylov"
    dmeta = export_model_for_aoti(_GMRES, "model", direct_out, derivatives=["stress:"])
    export_model_for_aoti(_GMRES_DERIV, "model", krylov_out, derivatives=["stress:"])
    # The direct baseline (krylov_gmres) keeps the default sensitivity solver,
    # which is direct DenseLU (the forward GMRES has no direct .solve).
    assert "solve_ift_package" in _implicit_seg(dmeta)

    ins = _inputs_from_meta(dmeta)
    _, Jd = Model(str(direct_out)).jacobian(ins)
    _, Jk = Model(str(krylov_out)).jacobian(ins)
    for o in Jd:
        for i in Jd[o]:
            assert torch.allclose(Jk[o][i], Jd[o][i], rtol=1e-6, atol=1e-8), (
                f"iterative vs direct input-sensitivity mismatch for ({o}, {i}): "
                f"max abs diff {(Jk[o][i] - Jd[o][i]).abs().max().item():.3e}"
            )


@_REQUIRES_PARAM_DERIV_TORCH
def test_param_sensitivity_krylov_metadata(tmp_path: Path):
    """An iterative param_sensitivity_solver emits `_dr_dparam` but NOT
    `_solve_param`; the segment records param_sensitivity.kind == krylov."""
    from neml2.cli.aoti_export import export_model_for_aoti

    meta = export_model_for_aoti(
        _PARAM_GMRES, "model", tmp_path / "art", promoted={_QNAME}, derivatives=[f"x:{_QNAME}"]
    )
    seg = _implicit_seg(meta)
    assert "dr_dparam_package" in seg
    assert "solve_param_package" not in seg
    assert seg["param_sensitivity"]["kind"] == "krylov"
    for pair in seg["param_jacobian_pairs"]:
        assert "row_offset" in pair and "col_offset" in pair


@_REQUIRES_PARAM_DERIV_TORCH
def test_param_sensitivity_krylov_matches_direct(tmp_path: Path):
    """The compiled iterative-ParamIFT parameter Jacobian equals the direct
    (DenseLU) one and finite differences on the promoted parameter."""
    from neml2.aoti._aoti import Model as PybindModel
    from neml2.cli.aoti_export import export_model_for_aoti

    direct_out = tmp_path / "direct"
    krylov_out = tmp_path / "krylov"
    export_model_for_aoti(
        _PARAM, "model", direct_out, promoted={_QNAME}, derivatives=[f"x:{_QNAME}"]
    )
    export_model_for_aoti(
        _PARAM_GMRES, "model", krylov_out, promoted={_QNAME}, derivatives=[f"x:{_QNAME}"]
    )
    md = PybindModel(str(direct_out))
    mk = PybindModel(str(krylov_out))

    b = 5
    ins = {
        "x": torch.zeros(b, dtype=torch.float64),
        "x~1": torch.linspace(1.0, 2.0, b, dtype=torch.float64),
        "t": torch.full((b,), 0.5, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
    }
    _, pjac_d = md.param_jacobian(ins)
    _, pjac_k = mk.param_jacobian(ins)
    block_d = pjac_d["x"][_QNAME]
    block_k = pjac_k["x"][_QNAME]
    assert torch.allclose(block_k, block_d, rtol=1e-6, atol=1e-8), (
        f"iterative vs direct param-sensitivity mismatch: "
        f"max abs diff {(block_k - block_d).abs().max().item():.3e}"
    )

    # Finite-difference ground truth on the iterative model's forward.
    w0 = float(mk.named_parameters()[_QNAME])
    h = 1e-5 * max(abs(w0), 1.0)

    def x_at(v):
        mk.named_parameters()[_QNAME].fill_(float(v))
        return mk.forward(ins)["x"].clone()

    fd = (x_at(w0 + h) - x_at(w0 - h)) / (2 * h)
    mk.named_parameters()[_QNAME].fill_(w0)
    rel = (block_k - fd).abs().max().item() / (fd.abs().max().item() + 1e-30)
    assert rel < 1e-5, f"iterative param_jacobian disagrees with FD (rel={rel:.2e})"
