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

"""NEML2_CAPTURE_SOLVE_FAILURE: the recoverable ConvergenceError optionally
carries a *failure context* for offline debugging -- a per-element convergence
mask and the best-effort iterate per unknown variable. The capture is opt-in
(the env var) and available on every route an offline replay can load: the
pure-Python (py-eager) model, the compiled non-substepping (cpp-aoti) model, and
the compiled substepping (cpp-aoti) model. A downstream host (e.g. MOOSE) dumps
the failing *inputs*; the user reloads them here, re-runs the model, and reads
the stuck state straight off the exception.

The scenario is a *mixed* batch -- one row that converges and one that cannot --
so the mask is genuinely per-element (the isolation the feature exists for), not
a trivially all-False mask. Across routes the payload is the same information
keyed by unknown name; the ONE deliberate difference is that py-eager surfaces
the iterate as a TYPED wrapper (an eager-mode enrichment) while cpp-aoti, which
has no typed wrappers at its boundary, surfaces a raw ``torch.Tensor``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import neml2
from neml2.aoti import ConvergenceError
from neml2.aoti import Model as AOTIModel
from neml2.cli.aoti_export import export_model_for_aoti

_ENV = "NEML2_CAPTURE_SOLVE_FAILURE"

# Row 0 easy (converges single-shot), row 1 impossible (a step so large it cannot
# converge single-shot, nor even at a low substepping cap).
_MIXED = {
    "x": [0.2, 0.2],
    "x~1": [0.2, 0.2],
    "t": [1.0, 1000.0],
    "t~1": [0.0, 0.0],
}
_B = 2


def _model_i(max_substepping_level: int, linear_solver: str = "lu") -> str:
    """A minimal NONLINEAR scalar implicit (Perzyna cubic rate + backward Euler):
    ``r(x) = x - x~1 - (t - t~1) * (max(x,0))^3``. A small step converges
    single-shot; a huge step overshoots to a non-finite residual and cannot. The
    substepping cap is parameterized so the same physics drives the eager (must be
    0 -- the eager runtime rejects substepping), non-substepping compiled (0), and
    substepping compiled (>0) routes.

    A ``ConstantExtrapolationPredictor`` warm-starts ``x`` from ``x~1``: the masked
    substep driver's per-span state carries only the givens, so without a
    predictor an unknown would seed to zero on the substepping route (and the easy
    row would then fail single-shot), whereas the eager / non-substepping routes
    warm-start from the input ``x`` -- the predictor makes the seed uniform."""
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
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 25
    linear_solver = '{linear_solver}'
  []
  [lu]
    type = DenseLU
  []
  [gmres]
    type = GMRES
  []
[]
[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'x'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
    max_substepping_level = {max_substepping_level}
  []
[]
"""


def _write_model(tmp: Path, max_substepping_level: int, linear_solver: str = "lu") -> Path:
    src = tmp / f"model_L{max_substepping_level}_{linear_solver}.i"
    src.write_text(_model_i(max_substepping_level, linear_solver))
    return src


def _mixed_tensors() -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in _MIXED.items()}


def _py_eager_fail(src: Path) -> ConvergenceError:
    """Drive the pure-Python model on the mixed batch into a (whole-batch)
    non-convergence and return the raised ConvergenceError."""
    m = neml2.load_model(str(src), "model").to(torch.float64)
    ins = _mixed_tensors()
    args = tuple(m.input_spec[n](ins[n]) for n in m.input_spec)
    with pytest.raises(ConvergenceError) as ei:
        m(*args)
    return ei.value


def _cpp_aoti_fail(out: Path, ins: dict[str, torch.Tensor] | None = None) -> ConvergenceError:
    """Mixed batch through a compiled model's forward, rooted at ``out``."""
    aoti = AOTIModel(str(out))
    with pytest.raises(ConvergenceError) as ei:
        aoti.forward(_mixed_tensors() if ins is None else ins)
    return ei.value


def _cpp_aoti_jacobian_fail(
    out: Path, ins: dict[str, torch.Tensor] | None = None
) -> ConvergenceError:
    """Mixed batch through a compiled model's Jacobian, rooted at ``out``."""
    aoti = AOTIModel(str(out))
    with pytest.raises(ConvergenceError) as ei:
        aoti.jacobian(_mixed_tensors() if ins is None else ins)
    return ei.value


def _mixed_2d_tensors() -> dict[str, torch.Tensor]:
    """A multi-axis (2, 2) dynamic batch mixing easy + impossible columns, so the
    driver flattens (2, 2) -> (4,), maxes out on the hard entries, and the capture
    context is reshaped back to (2, 2)."""
    row = {"x": [0.2, 0.2], "x~1": [0.2, 0.2], "t": [1.0, 1000.0], "t~1": [0.0, 0.0]}
    return {k: torch.tensor([v, v], dtype=torch.float64) for k, v in row.items()}


@pytest.fixture(scope="module")
def eager_src(tmp_path_factory) -> Path:
    """The eager runtime rejects substepping, so this model caps it at 0."""
    return _write_model(tmp_path_factory.mktemp("eager"), 0)


@pytest.fixture(scope="module")
def eager_gmres_src(tmp_path_factory) -> Path:
    """Same model with a matrix-free (GMRES) linear solver -- a masked re-solve is
    only available for direct solvers, so capture degrades to a bare error here."""
    return _write_model(tmp_path_factory.mktemp("eager_gmres"), 0, linear_solver="gmres")


@pytest.fixture(scope="module")
def nonsubstep_artifact(tmp_path_factory) -> Path:
    """Compiled model with substepping OFF: the hard row fails the single-shot
    solve, exercising the ``_run_implicit_segment`` capture path."""
    tmp = tmp_path_factory.mktemp("nl_nonsubstep")
    out = tmp / "nl"
    export_model_for_aoti(_write_model(tmp, 0), "model", out)
    return out


@pytest.fixture(scope="module")
def substep_artifact(tmp_path_factory) -> Path:
    """Compiled model with substepping ON but capped low, so the hard row maxes
    out substepping and raises -- exercising the masked-substep capture path.
    Exported with a derivative pair so both the value AND the substepped-Jacobian
    capture paths are reachable from this one artifact."""
    tmp = tmp_path_factory.mktemp("nl_substep")
    out = tmp / "nl"
    export_model_for_aoti(_write_model(tmp, 1), "model", out, derivatives=["x:t"])
    return out


def _assert_mixed_enriched(err, *, typed: bool, shape: tuple[int, ...] = (_B,)) -> None:
    """The error carries a genuinely *mixed* per-element convergence mask and the
    best-effort iterate keyed by unknown name, at the original dynamic-batch
    ``shape``. ``typed`` selects the py-eager (typed wrapper) vs cpp-aoti (raw
    tensor) surface."""
    mask = getattr(err, "converged_mask", None)
    assert mask is not None, "converged_mask not attached"
    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == shape
    # The isolation the feature exists for: some elements converged, some did not
    # -- NOT a trivially all-False mask (and not all-True either).
    assert mask.any(), "expected at least one element to have converged"
    assert not mask.all(), "expected at least one element to be unconverged"

    unknowns = getattr(err, "unknowns", None)
    assert unknowns is not None, "unknowns not attached"
    assert set(unknowns) == {"x"}, "the single unknown of the implicit model"
    w = unknowns["x"]
    if typed:
        # py-eager enrichment: a typed wrapper (NOT a raw torch.Tensor), whose
        # underlying data carries the dynamic-batch shape.
        assert not isinstance(w, torch.Tensor)
        assert isinstance(w.data, torch.Tensor)
        assert tuple(w.data.shape) == shape
    else:
        # cpp-aoti: a plain torch.Tensor at the boundary (no typed wrappers).
        assert isinstance(w, torch.Tensor)
        assert tuple(w.shape) == shape


# --- capture enriches the error, per route ---------------------------------


def test_py_eager_capture_enriches_error(monkeypatch, eager_src):
    monkeypatch.setenv(_ENV, "1")
    _assert_mixed_enriched(_py_eager_fail(eager_src), typed=True)


def test_cpp_aoti_nonsubstep_capture_enriches_error(monkeypatch, nonsubstep_artifact):
    monkeypatch.setenv(_ENV, "1")
    _assert_mixed_enriched(_cpp_aoti_fail(nonsubstep_artifact), typed=False)


def test_cpp_aoti_substep_capture_enriches_error(monkeypatch, substep_artifact):
    """The masked substepping driver, when it maxes out, attaches the level-0
    (single-shot) failure context -- the same information the non-substepping path
    reports -- so a substepping-compiled model is diagnosable too."""
    monkeypatch.setenv(_ENV, "1")
    _assert_mixed_enriched(_cpp_aoti_fail(substep_artifact), typed=False)


def test_cpp_aoti_substep_jacobian_capture_enriches_error(monkeypatch, substep_artifact):
    """The substepped-Jacobian driver, when it maxes out, attaches the same
    level-0 failure context as the value path -- so a failed consistent-tangent
    evaluation is diagnosable too, not just a failed forward."""
    monkeypatch.setenv(_ENV, "1")
    _assert_mixed_enriched(_cpp_aoti_jacobian_fail(substep_artifact), typed=False)


def test_cpp_aoti_substep_multiaxis_capture_enriches_error(monkeypatch, substep_artifact):
    """A multi-axis (2, 2) dynamic batch is flattened to 1-D inside the masked
    driver; on max-out the captured mask + iterate are reshaped back to (2, 2), so
    the failure context matches the caller's batch shape."""
    monkeypatch.setenv(_ENV, "1")
    err = _cpp_aoti_fail(substep_artifact, _mixed_2d_tensors())
    _assert_mixed_enriched(err, typed=False, shape=(2, 2))


def test_py_eager_iterative_solver_skips_capture(monkeypatch, eager_gmres_src):
    """Capture needs a masked re-solve, which only direct linear solvers support;
    with a matrix-free (GMRES) solver the masked path raises NotImplementedError,
    so capture is skipped and the error stays bare -- even with capture enabled."""
    monkeypatch.setenv(_ENV, "1")
    err = _py_eager_fail(eager_gmres_src)
    assert not hasattr(err, "converged_mask")
    assert not hasattr(err, "unknowns")


# --- opt-out leaves the error bare (byte-for-byte legacy behavior) ----------


def test_py_eager_opt_out_leaves_error_bare(monkeypatch, eager_src):
    monkeypatch.delenv(_ENV, raising=False)
    err = _py_eager_fail(eager_src)
    assert not hasattr(err, "converged_mask")
    assert not hasattr(err, "unknowns")
    assert isinstance(err, RuntimeError)  # unchanged: still a recoverable RuntimeError


def test_cpp_aoti_nonsubstep_opt_out_leaves_error_bare(monkeypatch, nonsubstep_artifact):
    monkeypatch.delenv(_ENV, raising=False)
    err = _cpp_aoti_fail(nonsubstep_artifact)
    assert not hasattr(err, "converged_mask")
    assert not hasattr(err, "unknowns")


def test_cpp_aoti_substep_opt_out_leaves_error_bare(monkeypatch, substep_artifact):
    monkeypatch.delenv(_ENV, raising=False)
    err = _cpp_aoti_fail(substep_artifact)
    assert not hasattr(err, "converged_mask")
    assert not hasattr(err, "unknowns")


# --- truthy / falsy env parsing (consistent C++ and Python) -----------------


@pytest.mark.parametrize("val", ["0", "false", "off", "no", "  ", "  OFF  "])
def test_capture_falsy_values_disable(monkeypatch, eager_src, val):
    """Unset / "" / 0 / false / no / off (case-insensitive, trimmed) are OFF."""
    monkeypatch.setenv(_ENV, val)
    err = _py_eager_fail(eager_src)
    assert not hasattr(err, "converged_mask")
    assert not hasattr(err, "unknowns")


@pytest.mark.parametrize("val", ["1", "true", "on", "yes", "  On  ", "anything"])
def test_capture_truthy_values_enable(monkeypatch, eager_src, val):
    """Any other set value (incl. 1 / true / yes / on) is ON."""
    monkeypatch.setenv(_ENV, val)
    err = _py_eager_fail(eager_src)
    assert getattr(err, "converged_mask", None) is not None
