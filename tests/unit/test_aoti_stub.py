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

"""Solver-discovery + minimal ``[Solvers]`` carry-over in ``emit_aoti_stub``.

``emit_aoti_stub`` (``neml2.cli.aoti_compile``) is purely structural -- it
parses the source HIT with ``nmhit`` and clones sections; it never instantiates
a model or solver. That lets these tests drive every solver-discovery branch
with tiny crafted inputs (the ``type`` strings need not be real registered
types):

* a direct ``ImplicitUpdate`` target -> its own solver is carried, the
  ``linear_solver`` field stripped (schema v4: solver config lives in the stub,
  the linear solver is baked into the compiled graph);
* a forward-only target with a stray ``[Solvers]`` block -> nothing carried,
  no ``solver`` field on the shim;
* a composed target over two differently-configured implicit segments -> the
  first solver is chosen and a disambiguation warning is emitted;
* a composed target over a single implicit segment whose solver has no
  ``linear_solver`` -> fallback selection, no warning, nothing to strip.
"""

from __future__ import annotations

from pathlib import Path

import nmhit
import pytest

from neml2.cli.aoti_compile import emit_aoti_stub


def _emit(tmp_path: Path, hit: str, model: str):
    """Write *hit*, run ``emit_aoti_stub``, return the re-parsed stub sections."""
    src = tmp_path / "in.i"
    src.write_text(hit)
    meta = tmp_path / "model_meta.json"
    stub = tmp_path / "stub.i"
    emit_aoti_stub(src, model, meta, stub)
    root = nmhit.parse_file(stub, [], [])
    return {s.path(): s for s in root.children(nmhit.NodeType.Section)}


def _shim(sections, model):
    return {s.path(): s for s in sections["Models"].children(nmhit.NodeType.Section)}[model]


def _solver_block_names(sections):
    if "Solvers" not in sections:
        return None
    return [s.path() for s in sections["Solvers"].children(nmhit.NodeType.Section)]


_DIRECT_IMPLICIT = """
[Models]
  [model]
    type = ImplicitUpdate
    solver = newton
  []
[]
[Solvers]
  [newton]
    type = Newton
    linear_solver = lu
  []
  [lu]
    type = DenseLU
  []
[]
"""

_FORWARD_ONLY = """
[Models]
  [model]
    type = LinearIsotropicElasticity
  []
[]
[Solvers]
  [newton]
    type = Newton
  []
[]
"""

_COMPOSED_TWO_SOLVERS = """
[Models]
  [top]
    type = ComposedModel
  []
  [eq1]
    type = ImplicitUpdate
    solver = n1
  []
  [eq2]
    type = ImplicitUpdate
    solver = n2
  []
[]
[Solvers]
  [n1]
    type = Newton
  []
  [n2]
    type = NewtonWithLineSearch
  []
[]
"""

_COMPOSED_ONE_SOLVER = """
[Models]
  [top]
    type = ComposedModel
  []
  [eq1]
    type = ImplicitUpdate
    solver = n1
  []
[]
[Solvers]
  [n1]
    type = Newton
  []
[]
"""

_COMPOSED_SHARED_SOLVER = """
[Models]
  [top]
    type = ComposedModel
  []
  [eq1]
    type = ImplicitUpdate
    solver = n1
  []
  [eq2]
    type = ImplicitUpdate
    solver = n1
  []
[]
[Solvers]
  [n1]
    type = Newton
  []
[]
"""


def test_direct_implicit_carries_solver_and_strips_linear_solver(tmp_path):
    sections = _emit(tmp_path, _DIRECT_IMPLICIT, "model")
    # Shim references the solver by name.
    assert _shim(sections, "model").param_optional_str("solver", "") == "newton"
    # Only the target solver is carried; the [lu] sub-solver is dropped.
    assert _solver_block_names(sections) == ["newton"]
    # The baked-in linear_solver field is stripped (editing it would be inert).
    newton = next(s for s in sections["Solvers"].children(nmhit.NodeType.Section))
    assert newton.find("linear_solver") is None


def test_forward_only_drops_solvers_block_and_omits_shim_field(tmp_path):
    sections = _emit(tmp_path, _FORWARD_ONLY, "model")
    # No implicit solver to carry -> no solver field, no [Solvers] block.
    assert _shim(sections, "model").param_optional_str("solver", "") == ""
    assert _solver_block_names(sections) is None


def test_multiple_implicit_solvers_warn_and_pick_first(tmp_path, capsys):
    sections = _emit(tmp_path, _COMPOSED_TWO_SOLVERS, "top")
    # Falls back to the first implicit solver in the graph.
    assert _shim(sections, "top").param_optional_str("solver", "") == "n1"
    assert _solver_block_names(sections) == ["n1"]
    # A disambiguation warning names both solvers and the chosen one.
    err = capsys.readouterr().err
    assert "multiple implicit solvers" in err
    assert "n1" in err and "n2" in err


def test_composed_single_solver_fallback_without_warning(tmp_path, capsys):
    sections = _emit(tmp_path, _COMPOSED_ONE_SOLVER, "top")
    # Single implicit solver -> chosen by fallback, no warning.
    assert _shim(sections, "top").param_optional_str("solver", "") == "n1"
    assert _solver_block_names(sections) == ["n1"]
    # No linear_solver to strip on this one; emission still succeeds.
    n1 = next(s for s in sections["Solvers"].children(nmhit.NodeType.Section))
    assert n1.find("linear_solver") is None
    assert "multiple implicit solvers" not in capsys.readouterr().err


def test_shared_solver_across_segments_deduplicated(tmp_path, capsys):
    """Two implicit segments naming the *same* solver collapse to one entry --
    no duplicate in the discovered list, single carried block, no warning."""
    sections = _emit(tmp_path, _COMPOSED_SHARED_SOLVER, "top")
    assert _shim(sections, "top").param_optional_str("solver", "") == "n1"
    assert _solver_block_names(sections) == ["n1"]
    assert "multiple implicit solvers" not in capsys.readouterr().err


def test_unknown_model_name_raises(tmp_path):
    with pytest.raises(ValueError, match="no \\[Models/nope\\] block"):
        _emit(tmp_path, _DIRECT_IMPLICIT, "nope")
