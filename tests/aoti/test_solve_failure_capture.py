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
(the env var) and identical across the two routes an offline replay can load:
the pure-Python (py-eager) model and the compiled (cpp-aoti) model. A downstream
host (e.g. MOOSE) dumps the failing *inputs*; the user reloads them here, re-runs
the model, and reads the stuck state straight off the exception -- see
framework/doc/content/source/neml2/actions/NEML2Action.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import neml2
from neml2.aoti import ConvergenceError
from neml2.aoti import Model as AOTIModel
from neml2.cli.aoti_export import export_model_for_aoti

_REPO = Path(__file__).resolve().parents[2]
# Minimal single-unknown ImplicitUpdate (scalar backward-Euler); miters=0 against
# a non-zero initial residual is a guaranteed, immediate non-convergence.
_IMPLICIT = _REPO / "tests" / "aoti" / "implicit_simple" / "model.i"
_ENV = "NEML2_CAPTURE_SOLVE_FAILURE"
_B = 4


def _py_eager_fail():
    """Drive the pure-Python model into a guaranteed non-convergence (miters=0)
    and return the raised ConvergenceError."""
    torch.manual_seed(0)
    m = neml2.load_model(str(_IMPLICIT), "model").to(torch.float64)
    m.solver.miters = 0
    args = tuple(m.input_spec[n](torch.rand(_B, dtype=torch.float64)) for n in m.input_spec)
    with pytest.raises(ConvergenceError) as ei:
        m(*args)
    return ei.value


def _cpp_aoti_fail(tmp_path: Path):
    """Same, through the compiled runtime: compile the model, override the baked
    solver config to miters=0, and drive it to non-convergence."""
    torch.manual_seed(0)
    py = neml2.load_model(str(_IMPLICIT), "model").to(torch.float64)
    names = list(py.input_spec)
    out = tmp_path / "impl"
    export_model_for_aoti(str(_IMPLICIT), "model", str(out))
    aoti = AOTIModel(str(out))
    aoti.set_solver_config(1e-10, 1e-8, 0, "none", 0, 0.5, 1e-4)  # miters=0
    ins = {n: torch.rand(_B, dtype=torch.float64) for n in names}
    with pytest.raises(ConvergenceError) as ei:
        aoti.forward(ins)
    return ei.value


def _assert_enriched(err):
    mask = getattr(err, "converged_mask", None)
    assert mask is not None, "converged_mask not attached"
    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == (_B,)
    # miters=0 -> nothing converged.
    assert not mask.any()
    unknown = getattr(err, "unknown", None)
    assert unknown is not None, "unknown not attached"
    assert set(unknown) == {"x"}  # the single unknown of the implicit_simple model
    # Best-effort iterate: a plain torch.Tensor (uniform across routes) with the
    # dynamic-batch leading dim.
    assert isinstance(unknown["x"], torch.Tensor)
    assert tuple(unknown["x"].shape) == (_B,)


def test_py_eager_capture_enriches_error(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    _assert_enriched(_py_eager_fail())


def test_cpp_aoti_capture_enriches_error(monkeypatch, tmp_path):
    monkeypatch.setenv(_ENV, "1")
    _assert_enriched(_cpp_aoti_fail(tmp_path))


def test_py_eager_opt_out_leaves_error_bare(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    err = _py_eager_fail()
    assert not hasattr(err, "converged_mask")
    assert not hasattr(err, "unknown")
    assert isinstance(err, RuntimeError)  # unchanged: still a recoverable RuntimeError


def test_cpp_aoti_opt_out_leaves_error_bare(monkeypatch, tmp_path):
    monkeypatch.delenv(_ENV, raising=False)
    err = _cpp_aoti_fail(tmp_path)
    assert not hasattr(err, "converged_mask")
    assert not hasattr(err, "unknown")


def test_capture_disabled_by_zero(monkeypatch):
    monkeypatch.setenv(_ENV, "0")  # explicit "0" is off
    err = _py_eager_fail()
    assert not hasattr(err, "converged_mask")
