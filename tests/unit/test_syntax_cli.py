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

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

from neml2.cli import syntax as _syntax_cli


def _record_map():
    return {record.type_name: record for record in _syntax_cli.collect_records()}


def test_collect_records_covers_native_sections():
    records = _record_map()

    assert records["LinearIsotropicElasticity"].section == "Models"
    assert records["ScalarLinearCombination"].section == "Models"
    assert records["Newton"].section == "Solvers"
    assert records["NonlinearSystem"].section == "EquationSystems"
    assert records["CubicCrystal"].section == "Data"
    # AOTIModel deliberately inherits from nn.Module rather than Model so the
    # bound AOTIModelPackageLoader runtime drives evaluation; the explicit
    # _MODEL_TYPES entry in cli/syntax.py keeps it visible under [Models].
    assert records["AOTIModel"].section == "Models"
    # The non-Model registered classes now expose HitSchema so the syntax
    # catalog and auto-doc pipeline can render their HIT surface.
    for type_name in (
        "TransientDriver",
        "TransientRegression",
        "Verification",
        "AOTIModel",
        "CSVScalar",
        "CSVSR2",
        "CSVVec",
        "CSVWR2",
    ):
        assert records[type_name].hit is not None, type_name


def test_hit_schema_record_emits_option_metadata():
    record = _record_map()["LinearIsotropicElasticity"]
    data = _syntax_cli.record_to_json(record)

    assert data["type"] == "LinearIsotropicElasticity"
    assert data["section"] == "Models"
    assert data["source_path"] == "models/solid_mechanics/elasticity/LinearIsotropicElasticity.py"
    assert [opt["name"] for opt in data["options"]] == [
        "strain",
        "stress",
        "compliance",
        "rate_form",
        "coefficients",
        "coefficient_types",
    ]
    assert data["options"][0]["ftype"] == "INPUT"
    assert data["options"][1]["ftype"] == "OUTPUT"
    assert data["options"][4]["required"] is True


def test_schema_backed_solver_emits_option_metadata():
    record = _record_map()["Newton"]
    data = _syntax_cli.record_to_json(record)

    assert data["section"] == "Solvers"
    assert {opt["name"] for opt in data["options"]} == {
        "linear_solver",
        "abs_tol",
        "rel_tol",
        "max_its",
    }


def test_derived_output_rename_knob_is_catalogued():
    """A ``derived_output``'s default name doubles as a rename knob (see
    ``schema._read_derived_var_name``); it must be discoverable as an OUTPUT
    option even though the field declares no explicit HIT option."""
    record = _record_map()["FredrickArmstrongPlasticHardening"]
    opts = {opt["name"]: opt for opt in _syntax_cli.record_to_json(record)["options"]}

    assert "back_stress_rate" in opts
    knob = opts["back_stress_rate"]
    assert knob["ftype"] == "OUTPUT"
    assert knob["required"] is False
    assert knob["type"] == "SR2"
    assert knob["doc"]  # synthesized, non-empty
    # The real inputs remain; the derived knob is additive.
    assert {"flow_rate", "flow_direction", "back_stress"} <= set(opts)


def test_derived_input_names_stay_hidden_but_residual_is_catalogued():
    """``derived_input`` names (previous-step ``~1``, rate) stay hidden, but a
    residual ``derived_output`` surfaces as a rename knob on the same model."""
    record = _record_map()["SR2BackwardEulerTimeIntegration"]
    names = {opt["name"] for opt in _syntax_cli.record_to_json(record)["options"]}

    assert "variable_residual" in names  # derived_output knob, surfaced
    assert "variable~1" not in names  # derived_input, hidden
    assert "variable_rate" not in names  # derived_input, hidden
    # The explicit user-facing options remain.
    assert {"variable", "time", "rate"} <= names


def test_collect_records_rejects_type_without_section(monkeypatch: pytest.MonkeyPatch):
    """A registered class missing ``SECTION`` is a programming bug — the
    catalog would silently drop the type otherwise. Fake a stale registry entry
    and assert :func:`collect_records` raises with a clear pointer at the
    offending class.
    """

    class _SectionlessType:
        """Stand-in for a registered class whose base forgot to declare SECTION."""

    monkeypatch.setitem(_syntax_cli._registry, "_SectionlessType", _SectionlessType)
    with pytest.raises(ValueError, match="no SECTION declared"):
        _syntax_cli.collect_records()


def test_record_without_schema_is_summary_only():
    record = _syntax_cli.SyntaxRecord(
        type_name="Handwritten",
        section="Solvers",
        doc="A record with no HitSchema.",
        source_path="solvers.py",
        class_name="",
        hit=None,
    )

    assert "options" not in _syntax_cli.record_to_json(record)


def test_emitted_docs_are_ascii_and_hit_options_are_documented():
    for record in _syntax_cli.collect_records():
        data = _syntax_cli.record_to_json(record)
        assert data["doc"].isascii(), data["type"]
        for option in data.get("options", []):
            assert option["doc"], (data["type"], option["name"])
            assert option["doc"].isascii(), (data["type"], option["name"], option["doc"])


def test_cli_writes_summary_to_stdout():
    stdout = io.StringIO()

    assert _syntax_cli.main(["--json", "-", "--summary", "--type", "Newton"], stdout=stdout) == 0
    data = json.loads(stdout.getvalue())

    assert len(data) == 1
    assert data[0]["type"] == "Newton"
    assert data[0]["section"] == "Solvers"
    assert "options" not in data[0]


def test_cli_writes_json_file(tmp_path: Path):
    output = tmp_path / "syntax.json"

    assert _syntax_cli.main(["--json", str(output), "--section", "Data"]) == 0
    data = json.loads(output.read_text())

    assert [record["type"] for record in data] == ["CubicCrystal"]


def test_server_methods():
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"id": 1, "method": "list_sections"}),
                json.dumps({"id": 2, "method": "list_types", "section": "Solvers"}),
                json.dumps({"id": 3, "method": "get_options", "type": "LinearIsotropicElasticity"}),
                json.dumps({"id": 4, "method": "get_options", "type": "MissingType"}),
                json.dumps({"id": 5, "method": "unknown"}),
                "{not json",
                "",
            ]
        )
    )
    stdout = io.StringIO()

    assert _syntax_cli.main(["--server"], stdin=stdin, stdout=stdout) == 0
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert responses[0] == {
        "id": 1,
        "result": ["Data", "Drivers", "EquationSystems", "Models", "Solvers", "Tensors"],
    }
    assert {record["type"] for record in responses[1]["result"]} >= {"Newton", "DenseLU"}
    assert responses[2]["result"]["options"][0]["name"] == "strain"
    assert responses[3] == {"id": 4, "result": None}
    assert responses[4] == {"id": 5, "error": "unknown method"}
    assert responses[5] == {"id": None, "error": "parse error"}


def test_server_rejects_other_flags():
    stderr = io.StringIO()

    assert _syntax_cli.main(["--server", "--summary"], stderr=stderr) == 1
    assert "--server is incompatible" in stderr.getvalue()


def test_generated_json_is_accepted_by_docs_extension():
    # Smoke-test the doc/_ext/neml2_syntax.py renderers against the live
    # output of `_syntax_cli.collect_records()`. Guards against
    # JSON-shape drift between the CLI and the doc extension.
    # Skip when sphinx isn't installed (the runtime test matrix doesn't
    # pull in the dev extras that ship sphinx).
    pytest.importorskip("sphinx")

    # Two levels up from tests/unit/test_syntax_cli.py to the repo root.
    repo = Path(__file__).resolve().parents[2]
    ext_path = repo / "doc/_ext/neml2_syntax.py"
    spec = importlib.util.spec_from_file_location("neml2_syntax", ext_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    records = [
        _syntax_cli.record_to_json(record, include_options=True)
        for record in _syntax_cli.collect_records()
    ]
    # Pick representative entries from a few sections — the per-type
    # renderer touches every code path (header, source, doc, options
    # grouped by ftype) without needing every type.
    samples = {}
    for r in records:
        sec = r.get("section") or ""
        if sec and sec not in samples:
            samples[sec] = r
    assert "Models" in samples and "Solvers" in samples

    for _sec, entry in samples.items():
        page = module._render_type_page(entry)
        assert entry["type"] in page
        assert page.startswith("(")  # MyST label
    # Path-grouped catalog: group Models by their source-tree submodule the
    # way doc/_ext/neml2_syntax.py:_generate does, then exercise the section
    # and submodule index renderers.
    models = [r for r in records if r.get("section") == "Models"]
    groups: dict[tuple, list] = {}
    for r in models:
        groups.setdefault(module._submodule_parts(r.get("source_path", "")), []).append(r)
    child_subs = sorted({parts[0] for parts in groups if parts})
    root_types = sorted(r["type"] for r in groups.get((), []))
    section_index = module._render_section_index("Models", child_subs, root_types)
    assert section_index.startswith("(models-syntax)=")
    assert child_subs and all(f"{sub}/index" in section_index for sub in child_subs)

    nested = next(parts for parts in groups if parts)  # e.g. ("solid_mechanics", ...)
    sub_index = module._render_submodule_index(
        "Models", nested, [], sorted(r["type"] for r in groups[nested])
    )
    assert sub_index.startswith(f"# {nested[-1]}")

    top = module._render_top_index(["Models", "Solvers"])
    assert "Models/index" in top and "Solvers/index" in top

    # Inert RST cross-reference roles are degraded to code spans (MyST can't
    # render them), so none should survive into a rendered type page.
    rst_roles = (":func:`", ":class:`", ":meth:`", ":attr:`")
    role_entry = next(
        (r for r in records if any(role in (r.get("doc") or "") for role in rst_roles)),
        None,
    )
    assert role_entry is not None, "expected at least one docstring with an RST role"
    role_page = module._render_type_page(role_entry)
    assert not any(role in role_page for role in rst_roles)
