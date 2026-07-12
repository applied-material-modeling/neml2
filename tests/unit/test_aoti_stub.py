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

"""Structural stub emission in ``emit_aoti_stub``.

``emit_aoti_stub`` (``neml2.cli.aoti_compile``) is purely structural -- it
parses the source HIT with ``nmhit`` and clones the sections it keeps; it never
instantiates a model or solver. That lets these tests drive its section-carry /
section-drop policy with tiny crafted inputs (the ``type`` strings need not be
real registered types).

Schema v10 collapsed the stub's solver plumbing: the artifact folder is
self-describing (one shared ``metadata.json`` at its root -- carrying structural
metadata *and* the implicit Newton's ``solver_config`` -- plus per-
``<device>/<dtype>/`` ``.pt2`` binaries), so the stub no longer carries a
``[Solvers]`` block or a ``solver`` field on the shim. The C++ ctor reads the
solver config straight from ``metadata.json`` at load (that round-trip is
covered by ``tests/unit/test_aoti_export.py``). The stub is reduced to:

* a single ``[Models]`` block holding only the ``AOTIModel`` shim -- ``type`` +
  an absolute ``artifact_path`` pointing at the artifact ROOT folder;
* ``[Settings]`` from the original (``aoti_*`` keys stripped);
* ``[Tensors]`` verbatim;
* ``[Drivers]`` verbatim, only when ``keep_drivers=True`` (``--driver`` mode).

``[Data]``, ``[EquationSystems]`` and ``[Solvers]`` are dropped in both modes.
"""

from __future__ import annotations

from pathlib import Path

import nmhit
import pytest

from neml2.cli.aoti_compile import emit_aoti_stub


def _emit(tmp_path: Path, hit: str, model: str, *, keep_drivers: bool = False):
    """Write *hit*, run ``emit_aoti_stub``, return the re-parsed stub sections."""
    src = tmp_path / "in.i"
    src.write_text(hit)
    artifact_dir = tmp_path / model
    stub = tmp_path / "stub.i"
    emit_aoti_stub(src, model, artifact_dir, stub, keep_drivers=keep_drivers)
    root = nmhit.parse_file(stub, [], [])
    return {s.path(): s for s in root.children(nmhit.NodeType.Section)}


def _shim(sections, model):
    return {s.path(): s for s in sections["Models"].children(nmhit.NodeType.Section)}[model]


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

_WITH_RUNTIME_SECTIONS = """
[Models]
  [model]
    type = ImplicitUpdate
    solver = newton
  []
[]
[Solvers]
  [newton]
    type = Newton
  []
[]
[Data]
  [d]
    type = SomeData
  []
[]
[EquationSystems]
  [es]
    type = SomeSystem
  []
[]
[Settings]
  aoti_mode = true
  aoti_device = cpu
  machine_precision = 1e-15
[]
[Tensors]
  [t]
    type = Scalar
    values = 1.0
  []
[]
[Drivers]
  [drv]
    type = TransientDriver
    model = 'model'
  []
[]
"""


def test_shim_has_only_type_and_artifact_path(tmp_path):
    """The shim carries exactly ``type = AOTIModel`` + an absolute
    ``artifact_path`` pointing at the artifact ROOT folder -- no ``solver`` field
    (v10: solver config lives in metadata.json) and no legacy ``meta`` field."""
    sections = _emit(tmp_path, _DIRECT_IMPLICIT, "model")
    shim = _shim(sections, "model")
    assert shim.param_str("type") == "AOTIModel"
    ap = shim.param_optional_str("artifact_path", "")
    assert ap and Path(ap).is_absolute()
    assert Path(ap).name == "model"  # the per-model artifact root folder
    # Superseded fields are gone.
    assert shim.find("solver") is None
    assert shim.find("meta") is None


def test_solvers_block_always_dropped(tmp_path):
    """No ``[Solvers]`` block is ever carried -- for an implicit target (whose
    solver config now rides in metadata.json) or a forward-only one."""
    for hit in (_DIRECT_IMPLICIT, _FORWARD_ONLY):
        sections = _emit(tmp_path, hit, "model")
        assert "Solvers" not in sections
        assert _shim(sections, "model").find("solver") is None


def test_forward_only_reduces_to_models_block(tmp_path):
    """A forward-only target with a stray ``[Solvers]`` block reduces to a lone
    ``[Models]`` block: no solver field, no ``[Solvers]``, nothing else."""
    sections = _emit(tmp_path, _FORWARD_ONLY, "model")
    assert list(sections) == ["Models"]
    assert _shim(sections, "model").find("solver") is None


def test_data_and_equationsystems_dropped(tmp_path):
    """Runtime-irrelevant ``[Data]`` / ``[EquationSystems]`` are dropped (their
    state was baked into the ``.pt2`` at compile time)."""
    sections = _emit(tmp_path, _WITH_RUNTIME_SECTIONS, "model")
    assert "Data" not in sections
    assert "EquationSystems" not in sections
    assert "Solvers" not in sections


def test_settings_carried_with_aoti_keys_stripped(tmp_path):
    """``[Settings]`` is carried, but the ``aoti_*`` keys (invalid for the v3
    factory) are stripped; other settings survive."""
    sections = _emit(tmp_path, _WITH_RUNTIME_SECTIONS, "model")
    assert "Settings" in sections
    settings = sections["Settings"]
    assert settings.find("aoti_mode") is None
    assert settings.find("aoti_device") is None
    assert settings.find("machine_precision") is not None


def test_tensors_carried_verbatim(tmp_path):
    """``[Tensors]`` is carried verbatim (a kept driver may reference any entry)."""
    sections = _emit(tmp_path, _WITH_RUNTIME_SECTIONS, "model")
    assert "Tensors" in sections
    names = [s.path() for s in sections["Tensors"].children(nmhit.NodeType.Section)]
    assert names == ["t"]


def test_drivers_dropped_in_model_mode(tmp_path):
    """``--model`` mode (``keep_drivers=False``) drops ``[Drivers]``."""
    sections = _emit(tmp_path, _WITH_RUNTIME_SECTIONS, "model", keep_drivers=False)
    assert "Drivers" not in sections


def test_drivers_kept_in_driver_mode(tmp_path):
    """``--driver`` mode (``keep_drivers=True``) carries ``[Drivers]`` verbatim."""
    sections = _emit(tmp_path, _WITH_RUNTIME_SECTIONS, "model", keep_drivers=True)
    assert "Drivers" in sections
    names = [s.path() for s in sections["Drivers"].children(nmhit.NodeType.Section)]
    assert names == ["drv"]


def test_unknown_model_name_raises(tmp_path):
    with pytest.raises(ValueError, match="no \\[Models/nope\\] block"):
        _emit(tmp_path, _DIRECT_IMPLICIT, "nope")
