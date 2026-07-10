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

"""AOTI (cpp-aoti via the pybind binding) substepping smoke tests.

These exercise the host-side substep driver through the compiled runtime end to
end: the Python role classification is serialized into the shared ``metadata.json``
(schema v10), the C++ ``Model`` parses it, and ``_run_implicit_segment_substepped``
dispatches per the roles. The scenario integration is linear-in-time, so the
substepped answer equals the single-shot answer -- the "no-op equivalence" that
validates the interpolation / chaining machinery. Genuine convergence-recovery
(a step that fails single shot but converges substepped) needs a nonlinear
residual and lands with the substepped-Jacobian work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.solvers import ConvergenceError

_REPO = Path(__file__).resolve().parents[2]
_SUBSTEP = _REPO / "tests" / "aoti" / "implicit_substep" / "model.i"
_SUBSTEP_NL = _REPO / "tests" / "aoti" / "implicit_substep_nl" / "model.i"


def _implicit_seg(meta: dict) -> dict:
    (seg,) = [s for s in meta["segments"] if s["kind"] == "implicit"]
    return seg


def test_metadata_carries_substep_config_and_roles(tmp_path: Path):
    """The exporter serializes the depth cap + per-given roles the C++ driver
    reads (single source of truth: the Python classifier)."""
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "implicit_substep"
    meta = export_model_for_aoti(_SUBSTEP, "model", out)

    seg = _implicit_seg(meta)
    assert seg["max_substepping_level"] == 2
    roles = {g["name"]: (g["role"], g.get("pair")) for g in seg["givens"]}
    assert roles["x~1"] == ("old_state", "x")
    assert roles["t"] == ("cur_force", "t~1")
    assert roles["t~1"] == ("old_force", "t")
    assert roles["x_rate"] == ("static", None)


def test_forward_noop_equivalence(tmp_path: Path):
    """Compiled substepped forward integrates to the analytical single-shot
    answer: x = x~1 + (t - t~1) * x_rate. Exercises driver dispatch + endpoint
    interpolation + chaining through the C++ runtime."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "implicit_substep"
    export_model_for_aoti(_SUBSTEP, "model", out)
    aoti = AOTIModel(str(out))

    b = 4
    inputs = {
        "x": torch.zeros(b, dtype=torch.float64),  # initial guess
        "x~1": torch.full((b,), 0.5, dtype=torch.float64),
        "t": torch.full((b,), 2.0, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
        "x_rate": torch.full((b,), 3.0, dtype=torch.float64),
    }
    outs = aoti.forward(inputs)
    # x = 0.5 + (2 - 0) * 3 = 6.5, regardless of how many sub-steps were taken.
    expected = torch.full((b,), 6.5, dtype=torch.float64)
    assert torch.allclose(outs["x"], expected, rtol=1e-10, atol=1e-10)


def test_forward_matches_non_substepped_build(tmp_path: Path):
    """A substepping build (level=2) and the plain implicit_simple build (level=0)
    produce identical forward output on the same easy step."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    plain_i = _REPO / "tests" / "aoti" / "implicit_simple" / "model.i"
    out_sub = tmp_path / "sub"
    out_plain = tmp_path / "plain"
    export_model_for_aoti(_SUBSTEP, "model", out_sub)
    export_model_for_aoti(plain_i, "model", out_plain)
    m_sub = AOTIModel(str(out_sub))
    m_plain = AOTIModel(str(out_plain))

    b = 3
    inputs = {
        "x": torch.zeros(b, dtype=torch.float64),
        "x~1": torch.linspace(0.0, 1.0, b, dtype=torch.float64),
        "t": torch.full((b,), 1.5, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
        "x_rate": torch.linspace(-1.0, 2.0, b, dtype=torch.float64),
    }
    a = m_sub.forward(inputs)["x"]
    c = m_plain.forward(inputs)["x"]
    assert torch.allclose(a, c, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Nonlinear model: genuine bisection recovery + chained consistent tangent.
# ---------------------------------------------------------------------------


def _eager_single_shot(x1: float, t_end: float, miters: int):
    """One eager single-shot solve of the nonlinear model over [0, t_end].
    Returns the solved x, or raises ConvergenceError if Newton can't converge in
    `miters` iterations (the contrast case substepping is meant to rescue)."""
    import neml2

    m = neml2.load_model(str(_SUBSTEP_NL), "model")
    m.max_substepping_level = 0  # true single shot -- disable substepping recovery
    m.solver.miters = miters
    ins = {"x": x1, "x~1": x1, "t": t_end, "t~1": 0.0}
    args = tuple(m.input_spec[n](torch.tensor([ins[n]], dtype=torch.float64)) for n in m.input_spec)
    return m(*args)[0].data.item()


def test_nonlinear_convergence_recovery(tmp_path: Path):
    """The core fix: a step that fails the single Newton solve (dt large vs
    max_its) converges via substepping, while on an easy dt the substepped result
    equals the eager single shot (no-op equivalence for a nonlinear model)."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    export_model_for_aoti(_SUBSTEP_NL, "model", out, derivatives=["x:x~1", "x:t", "x:t~1"])
    aoti = AOTIModel(str(out))

    def fwd(dt: float) -> float:
        ins = {
            "x": torch.full((1,), 0.2, dtype=torch.float64),
            "x~1": torch.full((1,), 0.2, dtype=torch.float64),
            "t": torch.full((1,), dt, dtype=torch.float64),
            "t~1": torch.zeros(1, dtype=torch.float64),
        }
        return aoti.forward(ins)["x"].item()

    # Contrast: dt=8 single-shot genuinely diverges (Newton overshoots the cubic
    # rate to a non-finite residual) even with a generous cap -- this is the
    # failure substepping rescues.
    with pytest.raises(ConvergenceError):
        _eager_single_shot(0.2, 8.0, miters=25)

    # Recovery: the compiled substepped model reaches a finite solution at dt=8.
    x8 = fwd(8.0)
    assert x8 == x8 and x8 > 0.2, f"substepped dt=8 did not recover sensibly: {x8}"

    # No-op equivalence (nonlinear, single span): on an easy dt the substepped
    # answer equals the eager single shot to solver tolerance.
    assert fwd(1.0) == pytest.approx(_eager_single_shot(0.2, 1.0, miters=50), rel=1e-8)


def test_nonlinear_substepped_jacobian_matches_fd(tmp_path: Path):
    """The chained consistent tangent (accumulated across bisected sub-steps)
    matches central finite differences of the substepped forward."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    export_model_for_aoti(_SUBSTEP_NL, "model", out, derivatives=["x:x~1", "x:t", "x:t~1"])
    aoti = AOTIModel(str(out))

    base = {
        "x": torch.full((1,), 0.2, dtype=torch.float64),
        "x~1": torch.full((1,), 0.2, dtype=torch.float64),
        "t": torch.full((1,), 8.0, dtype=torch.float64),
        "t~1": torch.zeros(1, dtype=torch.float64),
    }
    _, J = aoti.jacobian(base)

    h = 1e-6
    for name in ("x~1", "t", "t~1"):
        plus = {k: v.clone() for k, v in base.items()}
        minus = {k: v.clone() for k, v in base.items()}
        plus[name] = plus[name] + h
        minus[name] = minus[name] - h
        fd = (aoti.forward(plus)["x"] - aoti.forward(minus)["x"]) / (2 * h)
        ana = J["x"][name].reshape(fd.shape)
        assert torch.allclose(ana, fd, rtol=1e-5, atol=1e-6), (
            f"d x / d {name}: analytic {ana.item():.8e} vs FD {fd.item():.8e}"
        )


def test_eager_aoti_parity_easy_step(tmp_path: Path):
    """Cross-route parity on an EASY (single-span) step: the eager and cpp-aoti
    substepped forward + Jacobian agree. (On a hard step the adaptive *schedule*
    can differ between routes -- both produce a valid coarse solution -- so exact
    cross-route parity is only guaranteed where a single span converges.)"""
    import neml2
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    export_model_for_aoti(_SUBSTEP_NL, "model", out, derivatives=["x:x~1", "x:t", "x:t~1"])
    aoti = AOTIModel(str(out))
    eager = neml2.load_model(str(_SUBSTEP_NL), "model").to(torch.float64)

    ins = {
        "x": torch.full((1,), 0.2, dtype=torch.float64),
        "x~1": torch.full((1,), 0.2, dtype=torch.float64),
        "t": torch.full((1,), 1.0, dtype=torch.float64),  # easy: converges in one span
        "t~1": torch.zeros(1, dtype=torch.float64),
    }
    a_out, a_jac = aoti.jacobian(ins)
    e_out, e_jac = eager.jacobian(ins)
    assert torch.allclose(a_out["x"], e_out["x"].data, rtol=1e-9, atol=1e-9)
    for name in ("x~1", "t", "t~1"):
        assert torch.allclose(
            a_jac["x"][name].reshape(()), e_jac["x"][name].data.reshape(()), rtol=1e-8, atol=1e-9
        )


def test_masking_mixed_batch_matches_per_element(tmp_path: Path):
    """The masking payoff: in a batch mixing easy (dt small, 1 span) and hard
    (dt=8, deep bisection) rows, each row's substepped forward + Jacobian equals
    that row solved ALONE -- i.e. every element uses its own coarsest converging
    schedule and easy rows are not dragged to the hard row's schedule."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    export_model_for_aoti(_SUBSTEP_NL, "model", out, derivatives=["x:x~1", "x:t", "x:t~1"])
    aoti = AOTIModel(str(out))

    dts = [1.0, 8.0, 2.0, 8.0]  # easy, hard, easy, hard
    x1s = [0.2, 0.2, 0.3, 0.15]
    b = len(dts)

    def make(dt_list, x1_list):
        return {
            "x": torch.tensor(x1_list, dtype=torch.float64),
            "x~1": torch.tensor(x1_list, dtype=torch.float64),
            "t": torch.tensor(dt_list, dtype=torch.float64),
            "t~1": torch.zeros(len(dt_list), dtype=torch.float64),
        }

    out_b, jac_b = aoti.jacobian(make(dts, x1s))

    for i in range(b):
        out_i, jac_i = aoti.jacobian(make([dts[i]], [x1s[i]]))
        assert out_b["x"][i].item() == pytest.approx(out_i["x"][0].item(), rel=1e-12), (
            f"row {i} (dt={dts[i]}) forward differs from solved-alone"
        )
        for name in ("x~1", "t", "t~1"):
            assert jac_b["x"][name][i].item() == pytest.approx(
                jac_i["x"][name][0].item(), rel=1e-10, abs=1e-12
            ), f"row {i} (dt={dts[i]}) d x/d {name} differs from solved-alone"


def _mixed_inputs():
    # one easy row (dt=1, 1 span) + one hard row (dt=8, deep bisection).
    return {
        "x": torch.tensor([0.2, 0.2], dtype=torch.float64),
        "x~1": torch.tensor([0.2, 0.2], dtype=torch.float64),
        "t": torch.tensor([1.0, 8.0], dtype=torch.float64),
        "t~1": torch.zeros(2, dtype=torch.float64),
    }


def test_substep_trace_env_var(tmp_path: Path, capfd, monkeypatch):
    """NEML2_AOTI_TRACE_SUBSTEP=1 prints a per-solve substep summary to stderr
    (how many elements substepped, max depth, segment-solve count)."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    export_model_for_aoti(_SUBSTEP_NL, "model", out)
    aoti = AOTIModel(str(out))

    monkeypatch.setenv("NEML2_AOTI_TRACE_SUBSTEP", "1")
    aoti.forward(_mixed_inputs())
    err = capfd.readouterr().err
    assert "[aoti substep] value:" in err
    assert "1 substepped" in err  # exactly the one hard row
