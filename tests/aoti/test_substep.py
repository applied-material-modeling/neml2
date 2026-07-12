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
end: the Python role classification is serialized into the shared ``metadata.json``,
the C++ ``Model`` parses it, and ``_run_implicit_segment_substepped``
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
    import os
    from contextlib import contextmanager

    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    plain_i = _REPO / "tests" / "aoti" / "implicit_simple" / "model.i"
    out_sub = tmp_path / "sub"
    out_plain = tmp_path / "plain"
    export_model_for_aoti(_SUBSTEP, "model", out_sub)
    export_model_for_aoti(plain_i, "model", out_plain)

    b = 3
    inputs = {
        "x": torch.zeros(b, dtype=torch.float64),
        "x~1": torch.linspace(0.0, 1.0, b, dtype=torch.float64),
        "t": torch.full((b,), 1.5, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
        "x_rate": torch.linspace(-1.0, 2.0, b, dtype=torch.float64),
    }

    # The substepping and plain builds share an identical residual graph
    # (substepping is runtime-only), so their `.pt2` segments hash the same. On
    # Windows torch extracts each `.pt2` to a system-temp path keyed by that hash
    # and keeps the loaded `.pyd` locked for the process lifetime, so loading both
    # under one temp collides (sharing violation). Load each under its own
    # extraction temp so the identical segments land in distinct roots. No-op-ish
    # elsewhere (the override just points torch at a fresh scratch dir).
    @contextmanager
    def _own_extract_temp(name):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        saved = {k: os.environ.get(k) for k in ("TMP", "TEMP")}
        os.environ["TMP"] = os.environ["TEMP"] = str(d)
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    with _own_extract_temp("ex_sub"):
        a = AOTIModel(str(out_sub)).forward(inputs)["x"]
    with _own_extract_temp("ex_plain"):
        c = AOTIModel(str(out_plain)).forward(inputs)["x"]
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
    """Cross-route parity on an EASY (single-span) step: the cpp-aoti substepped
    forward + Jacobian equal the eager SINGLE-SHOT result. Substepping is an
    AOTI-only feature (the eager runtime rejects it), so the eager reference runs
    at level 0; on an easy step the compiled substep driver takes a single span,
    so the two agree. (On a hard step the compiled route substeps and there is no
    eager counterpart to compare against.)"""
    import neml2
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    export_model_for_aoti(_SUBSTEP_NL, "model", out, derivatives=["x:x~1", "x:t", "x:t~1"])
    aoti = AOTIModel(str(out))
    eager = neml2.load_model(str(_SUBSTEP_NL), "model").to(torch.float64)
    eager.max_substepping_level = 0  # eager does not substep; single-shot reference

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


def test_masking_maxed_out_raises_recoverable(tmp_path: Path):
    """Exhausting ``max_substepping_level`` in the MASKED cpp-aoti path must raise
    the RECOVERABLE ``ConvergenceError`` (so a time-stepper like MOOSE cuts the
    outer step and retries), NOT a fatal error.

    Regression: the masked driver used to signal max-out via ``_assert`` (which
    throws the non-recoverable ``FatalError``), so a maxed-out substep was
    unrecoverable downstream -- unlike the whole-batch driver, which re-throws the
    recoverable ``ConvergenceError``.
    """
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    # Cap substepping low so an extreme row cannot recover even at max depth.
    export_model_for_aoti(
        _SUBSTEP_NL,
        "model",
        out,
        additional_args=("Models/model/max_substepping_level:=1",),
    )
    aoti = AOTIModel(str(out))
    # Mixed 1-D batch (-> masked driver): one easy row + one row so hard it
    # exhausts the level-1 substepping.
    ins = {
        "x": torch.tensor([0.2, 0.2], dtype=torch.float64),
        "x~1": torch.tensor([0.2, 0.2], dtype=torch.float64),
        "t": torch.tensor([1.0, 1000.0], dtype=torch.float64),
        "t~1": torch.zeros(2, dtype=torch.float64),
    }
    with pytest.raises(ConvergenceError):
        aoti.forward(ins)


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


def test_multiaxis_batch_substepping_forward(tmp_path: Path):
    """A multi-axis (non-1-D) dynamic batch cannot use per-element masking (that
    requires a single leading batch axis), so it takes the whole-batch
    ``_run_implicit_segment_substepped`` driver. On a hard step it bisects and
    recovers the same finite solution the 1-D masked path reaches."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    export_model_for_aoti(_SUBSTEP_NL, "model", out)
    aoti = AOTIModel(str(out))

    def mk(v):
        return torch.full((2, 2), v, dtype=torch.float64)

    ins2d = {"x": mk(0.2), "x~1": mk(0.2), "t": mk(8.0), "t~1": mk(0.0)}
    out2d = aoti.forward(ins2d)["x"]
    assert out2d.shape == (2, 2)
    assert torch.isfinite(out2d).all() and (out2d > 0.2).all()

    # Same hard step through the 1-D (masked) path -> identical recovered value.
    ins1d = {k: v.reshape(-1)[:1] for k, v in ins2d.items()}
    out1d = aoti.forward(ins1d)["x"]
    assert torch.allclose(out2d, out1d.expand_as(out2d), rtol=1e-8, atol=1e-8)


def test_multiaxis_batch_substepped_jacobian_not_implemented(tmp_path: Path):
    """The whole-batch substepped Jacobian enters the non-masked driver and runs
    the substepped forward, but the underlying per-block (sub-batch) IFT Jacobian
    is not implemented -- so a multi-axis batch Jacobian raises a clear error
    rather than returning a wrong tangent. Pins that contract (and the driver's
    forward-then-compose entry path)."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "nl"
    export_model_for_aoti(_SUBSTEP_NL, "model", out, derivatives=["x:t"])
    aoti = AOTIModel(str(out))

    def mk(v):
        return torch.full((2, 2), v, dtype=torch.float64)

    ins2d = {"x": mk(0.2), "x~1": mk(0.2), "t": mk(8.0), "t~1": mk(0.0)}
    with pytest.raises(RuntimeError, match="not yet implemented|BLOCK"):
        aoti.jacobian(ins2d)


# ---------------------------------------------------------------------------
# Implicit-solve setup parity: warm start (no predictor) + line search.
# ---------------------------------------------------------------------------


def _scalar_implicit_i(*, solver: str = "Newton", predictor: bool = False, ls_type=None) -> str:
    """A minimal nonlinear scalar implicit (Perzyna cubic rate + backward Euler)
    with a configurable solver / predictor, for exercising the compiled implicit
    solve's setup paths (unknown warm start, Newton line search)."""
    predblk = (
        "  [predictor]\n    type = ConstantExtrapolationPredictor\n"
        "    unknowns_Scalar = 'x'\n  []\n"
        if predictor
        else ""
    )
    predref = "    predictor = 'predictor'\n" if predictor else ""
    ls = f"    linesearch_type = {ls_type}\n    max_linesearch_iterations = 4\n" if ls_type else ""
    return f"""
[Models]
  [rate]
    type = PerzynaPlasticFlowRate
    yield_function = 'x'
    flow_rate = 'x_rate'
    reference_stress = 1.0
    exponent = 3
  []
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
  [residual_model]
    type = ComposedModel
    models = 'rate integrate'
  []
[]
[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'residual_model'
    unknowns = 'x'
    residuals = 'x_residual'
  []
[]
[Solvers]
  [newton]
    type = {solver}
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 25
{ls}    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]
[Models]
{predblk}  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
{predref}  []
[]
"""


def _scalar_implicit_inputs():
    return {
        "x": torch.full((2,), 0.2, dtype=torch.float64),
        "x~1": torch.full((2,), 0.2, dtype=torch.float64),
        "t": torch.full((2,), 1.0, dtype=torch.float64),
        "t~1": torch.zeros(2, dtype=torch.float64),
    }


def _eager_forward(src: Path, ins: dict) -> torch.Tensor:
    import neml2

    eager = neml2.load_model(str(src), "model").to(torch.float64)
    args = tuple(eager.input_spec[n](ins[n]) for n in eager.input_spec)
    return eager(*args)[0].data


def test_no_predictor_warmstart_matches_eager(tmp_path: Path):
    """Without a predictor the compiled implicit solve must warm-start the unknown
    from its input value -- as eager's ``ImplicitUpdate._initial_unknowns`` does --
    not from zeros. Seeding zeros diverged the Newton on this stiff step (Newton
    overshoots the cubic rate to a non-finite residual) where eager converges: a
    py-eager <-> cpp-aoti parity violation. Regression for that fix."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    src = tmp_path / "np.i"
    src.write_text(_scalar_implicit_i(solver="Newton", predictor=False))
    out = tmp_path / "np"
    export_model_for_aoti(src, "model", out)
    ins = _scalar_implicit_inputs()
    a = AOTIModel(str(out)).forward(ins)["x"]
    assert torch.isfinite(a).all()
    e = _eager_forward(src, ins)
    assert torch.allclose(a, e, rtol=1e-9, atol=1e-9), f"aoti {a.tolist()} vs eager {e.tolist()}"


@pytest.mark.parametrize("ls_type", ["BACKTRACKING", "STRONG_WOLFE"])
def test_newton_linesearch_solver_path(tmp_path: Path, ls_type: str):
    """A NewtonWithLineSearch solver drives the compiled Newton's line-search
    branch (ls_max_iters > 1) -- the trial loop the default single-step Newton
    skips. The solver config rides in the artifact metadata; the result matches
    the eager route (both run the same shared C++ Newton)."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    src = tmp_path / f"ls_{ls_type}.i"
    src.write_text(_scalar_implicit_i(solver="NewtonWithLineSearch", ls_type=ls_type))
    out = tmp_path / f"ls_{ls_type}"
    export_model_for_aoti(src, "model", out)
    ins = _scalar_implicit_inputs()
    a = AOTIModel(str(out)).forward(ins)["x"]
    assert torch.isfinite(a).all() and (a > 0.2).all()
    e = _eager_forward(src, ins)
    assert torch.allclose(a, e, rtol=1e-9, atol=1e-9)


def test_newton_trace_verbose_logs(tmp_path: Path, monkeypatch):
    """``NEML2_AOTI_TRACE_NEWTON`` drives the compiled Newton's verbose
    convergence logging -- the per-iteration and per-line-search-iteration stderr
    trace MOOSE users rely on to inspect a solve. Level 2 with a line-search
    solver exercises both log blocks."""
    monkeypatch.setenv("NEML2_AOTI_TRACE_NEWTON", "2")
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    src = tmp_path / "trace.i"
    src.write_text(
        _scalar_implicit_i(solver="NewtonWithLineSearch", ls_type="BACKTRACKING", predictor=True)
    )
    out = tmp_path / "trace"
    export_model_for_aoti(src, "model", out)
    x = AOTIModel(str(out)).forward(_scalar_implicit_inputs())["x"]
    assert torch.isfinite(x).all()
