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

"""Tests for ``neml2-compile --interactive``.

The bulk of the coverage targets the framework-free pure helpers in
:mod:`neml2.cli._compile_interactive` -- importing that module never requires
``questionary``. The end-to-end flow test (with a scripted, fake ``questionary``)
and the ``neml2-compile -i`` argument-handling tests are guarded by
``importorskip("questionary")``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neml2.cli._compile_interactive import (
    FormState,
    IntrospectionForm,
    _dedupe,
    build_compile_argv,
    introspection_to_form,
    list_driver_names,
    list_model_names,
    preview_command,
    validate_state,
)

# The shared CLI worked-example input: a `LinearIsotropicElasticity` model named
# `elasticity` driven by a `TransientDriver` named `driver`.
_FIXTURE = Path(__file__).resolve().parents[2] / "doc" / "content" / "deployment" / "input.i"


# --------------------------------------------------------------------------- #
# build_compile_argv + the parse-back round-trip                              #
# --------------------------------------------------------------------------- #


def test_argv_minimal_model():
    state = FormState(input_file="input.i", target="model", name="elasticity", devices=("cpu",))
    assert build_compile_argv(state) == [
        "input.i",
        "--model",
        "elasticity",
        "--output-dir",
        "./aoti",
        "--device",
        "cpu",
        "--dtype",
        "float64",
    ]


def test_argv_driver_target():
    state = FormState(input_file="input.i", target="driver", name="driver", devices=("cpu",))
    assert build_compile_argv(state)[:3] == ["input.i", "--driver", "driver"]


def test_argv_multidevice_dedup_single_flag():
    state = FormState(name="m", devices=("cpu", "cuda", "cpu"))
    argv = build_compile_argv(state)
    i = argv.index("--device")
    # Single nargs="+" flag, de-duped, order preserved.
    assert argv[i : i + 3] == ["--device", "cpu", "cuda"]
    assert "--device" not in argv[i + 1 :]


def test_argv_promoted_and_derivatives_repeat():
    state = FormState(
        name="m",
        devices=("cpu",),
        promoted=("E", "nu"),
        derivatives=("stress:strain", ":"),
    )
    argv = build_compile_argv(state)
    assert [argv[k + 1] for k, t in enumerate(argv) if t == "-p"] == ["E", "nu"]
    assert [argv[k + 1] for k, t in enumerate(argv) if t == "-d"] == ["stress:strain", ":"]


def test_argv_renames_repeat():
    state = FormState(
        name="m",
        devices=("cpu",),
        rename_inputs=("strain:eps",),
        rename_outputs=("stress:sig",),
        rename_parameters=("elasticity.E:youngs", "elasticity.nu:poisson"),
    )
    argv = build_compile_argv(state)
    assert [argv[k + 1] for k, t in enumerate(argv) if t == "--rename-input"] == ["strain:eps"]
    assert [argv[k + 1] for k, t in enumerate(argv) if t == "--rename-output"] == ["stress:sig"]
    assert [argv[k + 1] for k, t in enumerate(argv) if t == "--rename-parameter"] == [
        "elasticity.E:youngs",
        "elasticity.nu:poisson",
    ]


def test_argv_example_shape_and_dynamic_batch_tristate():
    def mk(dynamic_batch: bool | None) -> FormState:
        return FormState(
            name="m",
            devices=("cpu",),
            example_shape=("strain=(2;100)",),
            dynamic_batch=dynamic_batch,
        )

    assert "--example-batch-shape" in build_compile_argv(mk(None))
    assert "--dynamic-batch" in build_compile_argv(mk(True))
    assert "--no-dynamic-batch" in build_compile_argv(mk(False))
    none_argv = build_compile_argv(mk(None))
    assert "--dynamic-batch" not in none_argv and "--no-dynamic-batch" not in none_argv


def test_argv_load_repeat_and_empty_output_dir_omitted():
    state = FormState(name="m", devices=("cpu",), output_dir="", load=("a.py", "b.py"))
    argv = build_compile_argv(state)
    assert "--output-dir" not in argv
    assert [argv[k + 1] for k, t in enumerate(argv) if t == "--load"] == ["a.py", "b.py"]


def test_argv_jobs_emitted_only_when_parallel():
    # jobs == 1 (the serial default) is omitted to keep the previewed command clean.
    assert "-j" not in build_compile_argv(FormState(name="m", devices=("cpu",), jobs=1))
    # jobs > 1 emits `-j N`.
    argv = build_compile_argv(FormState(name="m", devices=("cpu",), jobs=4))
    assert argv[argv.index("-j") + 1] == "4"


@pytest.mark.parametrize(
    "state",
    [
        FormState(input_file="input.i", target="model", name="elasticity", devices=("cpu",)),
        FormState(
            input_file="in.i",
            target="driver",
            name="driver",
            devices=("cpu", "cuda"),
            dtype="float32",
            output_dir="out",
            promoted=("E", "nu"),
            derivatives=("stress:strain",),
            example_shape=("strain=(2;100)",),
            dynamic_batch=False,
            jobs=3,
            load=("ext.py",),
        ),
        # Model target with boundary renames -- the built argv must parse cleanly.
        FormState(
            input_file="in.i",
            target="model",
            name="elasticity",
            devices=("cpu",),
            promoted=("elasticity.E",),
            rename_inputs=("strain:eps",),
            rename_outputs=("stress:sig",),
            rename_parameters=("elasticity.E:youngs",),
        ),
    ],
)
def test_argv_round_trips_through_real_parser(state):
    """The built argv must parse cleanly with neml2-compile's own parser, with no
    leftover tokens -- this is the guarantee that the preview == what the CLI runs."""
    from neml2.cli.aoti_compile import _build_arg_parser  # noqa: PLC2701, PLC0415

    argv = build_compile_argv(state)
    args, extra = _build_arg_parser().parse_known_args(argv)
    assert extra == []
    assert (args.model, args.driver)[state.target == "driver"] == state.name
    assert args.device == list(_dedupe(state.devices))
    assert args.dtype == state.dtype
    assert args.parameter == list(state.promoted)
    assert args.derivative == list(state.derivatives)
    assert args.example_batch_shape == list(state.example_shape)
    assert args.dynamic_batch is state.dynamic_batch
    assert args.jobs == state.jobs
    assert args.load == list(state.load)


def test_preview_quotes_shell_metacharacters():
    state = FormState(input_file="input.i", name="m", example_shape=("strain=(2;100)",))
    preview = preview_command(state)
    assert preview.startswith("neml2-compile ")
    # The semicolon would otherwise be a shell statement separator.
    assert "'strain=(2;100)'" in preview


# --------------------------------------------------------------------------- #
# validate_state                                                              #
# --------------------------------------------------------------------------- #


def test_validate_flags_missing_fields():
    problems = validate_state(FormState(input_file="", name="", devices=()))
    assert any("Input file" in p for p in problems)
    assert any("name" in p for p in problems)
    assert any("device" in p for p in problems)


def test_validate_flags_nonexistent_input(tmp_path):
    problems = validate_state(
        FormState(input_file=str(tmp_path / "nope.i"), name="m", devices=("cpu",))
    )
    assert any("not found" in p for p in problems)


def test_validate_flags_bad_jobs(tmp_path):
    inp = tmp_path / "in.i"
    inp.write_text("")
    problems = validate_state(FormState(input_file=str(inp), name="m", devices=("cpu",), jobs=0))
    assert any("-j" in p or "processes" in p.lower() for p in problems)


# --------------------------------------------------------------------------- #
# plan_summary -- the segment / artifact preview shown before choosing -j      #
# --------------------------------------------------------------------------- #

_TESTS_ROOT = Path(__file__).parent.parent
_ELASTICITY_I = _TESTS_ROOT / "models/solid_mechanics/elasticity/LinearIsotropicElasticity.i"
_COMPOSED_I = _TESTS_ROOT / "aoti/composed_param/model.i"


def test_plan_summary_single_segment_forward():
    import neml2  # noqa: F401 — registers models
    from neml2.cli._compile_interactive import plan_summary

    summary = plan_summary(
        FormState(input_file=str(_ELASTICITY_I), target="model", name="model", devices=("cpu",))
    )
    # A forward-only model is one segment; the last generated file is the stub.
    assert [(i, k) for (i, k, _b, _a) in summary.segments] == [(0, "forward")]
    assert summary.files[-1] == "model_aoti.i"
    assert "model_meta.json" in summary.files


def test_plan_summary_composed_multi_segment():
    import neml2  # noqa: F401 — registers models
    from neml2.cli._compile_interactive import plan_summary

    summary = plan_summary(
        FormState(
            input_file=str(_COMPOSED_I),
            target="model",
            name="model",
            devices=("cpu",),
            derivatives=(":",),
        )
    )
    assert [(i, k) for (i, k, _b, _a) in summary.segments] == [
        (0, "forward"),
        (1, "implicit"),
        (2, "forward"),
    ]
    # Every predicted artifact plus the meta + stub is enumerated, stub last.
    assert summary.files[-1] == "model_aoti.i"
    assert "model_seg1_ift.pt2" in summary.files


def test_validate_ok_for_real_file():
    assert (
        validate_state(FormState(input_file=str(_FIXTURE), name="elasticity", devices=("cpu",)))
        == []
    )


# --------------------------------------------------------------------------- #
# introspection_to_form                                                       #
# --------------------------------------------------------------------------- #


def test_introspection_to_form_reduces_model_dict():
    data = {
        "name": "Foo",
        "inputs": [{"name": "strain", "type": "SR2"}],
        "outputs": [{"name": "stress", "type": "SR2"}],
        "parameters": [{"name": "E"}, {"name": "nu"}],
        "buffers": [{"name": "buf0"}],
    }
    form = introspection_to_form(data)
    assert form.promotable == ["E", "nu", "buf0"]  # parameters then buffers
    assert form.inputs == ["strain"]
    assert form.outputs == ["stress"]
    assert form.input_types == {"strain": "SR2"}  # used for the example-shape hint


def test_introspection_to_form_empty():
    assert introspection_to_form({}) == IntrospectionForm()


def test_dedupe_preserves_order():
    assert _dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# HIT block listing (nmhit-only, no model instantiation)                      #
# --------------------------------------------------------------------------- #


def test_list_model_and_driver_names():
    assert list_model_names(_FIXTURE) == ["elasticity"]
    assert list_driver_names(_FIXTURE) == ["driver"]


def test_list_section_entries_includes_type():
    from neml2.cli._compile_interactive import list_section_entries  # noqa: PLC2701, PLC0415

    assert list_section_entries(_FIXTURE, "Models") == [("elasticity", "LinearIsotropicElasticity")]
    assert list_section_entries(_FIXTURE, "Drivers") == [("driver", "TransientDriver")]


# --------------------------------------------------------------------------- #
# Model-backed introspection + the real validators                           #
# --------------------------------------------------------------------------- #


def test_introspection_matches_inspect_dict():
    from neml2.cli.inspect import _model_to_dict  # noqa: PLC2701, PLC0415
    from neml2.factory import load_model  # noqa: PLC0415

    data = _model_to_dict(load_model(str(_FIXTURE), "elasticity"))
    form = introspection_to_form(data)
    assert form.outputs == [o["name"] for o in data["outputs"]]
    assert form.inputs == [i["name"] for i in data["inputs"]]
    assert form.promotable == [p["name"] for p in data["parameters"]] + [
        b["name"] for b in data["buffers"]
    ]
    # A non-trivial model has at least one output and one structural input.
    assert form.outputs and form.inputs


def test_builder_derivative_specs_accepted_by_real_resolver():
    """A derivative spec the form would emit must satisfy neml2-compile's own
    OUT:IN resolver; a bogus one must be rejected."""
    from neml2.cli.aoti_export import _resolve_derivative_specs  # noqa: PLC2701, PLC0415
    from neml2.factory import load_model  # noqa: PLC0415

    model = load_model(str(_FIXTURE), "elasticity")
    out = next(iter(model.output_spec))
    in_ = next(iter(model.input_spec))

    state = FormState(name="elasticity", devices=("cpu",), derivatives=(f"{out}:{in_}",))
    # Pull the -d spec straight out of the argv the form produced.
    argv = build_compile_argv(state)
    d_specs = [argv[k + 1] for k, t in enumerate(argv) if t == "-d"]
    struct, _ = _resolve_derivative_specs(d_specs, list(model.output_spec), list(model.input_spec))
    assert (out, in_) in struct

    with pytest.raises(ValueError):
        _resolve_derivative_specs(["stress:bogus_input_xyz"], list(model.output_spec), [in_])


# --------------------------------------------------------------------------- #
# End-to-end wizard flow (scripted fake questionary)                          #
# --------------------------------------------------------------------------- #


class _ScriptedPrompt:
    """Stand-in for a questionary prompt object: ``.ask()`` pops the next answer."""

    def __init__(self, answers):
        self._answers = answers

    def ask(self):
        return next(self._answers)


class _FakeQuestionary:
    """A fake ``questionary`` module that answers every prompt from a script, in
    call order, so the wizard flow can be driven without a TTY."""

    def __init__(self, answers):
        self._answers = iter(answers)

    @staticmethod
    def Choice(title, value=None, checked=False):  # noqa: N802 (mirror questionary API)
        return value if value is not None else title

    def print(self, *args, **kwargs):
        pass

    def _prompt(self, *args, **kwargs):
        return _ScriptedPrompt(self._answers)

    text = path = select = checkbox = confirm = _prompt


def test_wizard_flow_builds_and_runs_expected_argv(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # Prompt answers in flow order (input file comes from the CLI, not a prompt):
    # target, name, promote, derivatives?, rename?, devices, dtype, output dir,
    # uniform example shape?, uniform value, dynamic batch, then the review action.
    fake = _FakeQuestionary(
        [
            "model",
            "elasticity",
            ["E"],
            False,
            False,  # rename any boundary variables?
            ["cpu", "cuda"],
            "float32",
            "out",
            True,  # uniform example shape?
            "",  # uniform value (blank -> none)
            True,  # dynamic batch
            "run",  # review menu -> Run the compile
        ]
    )
    monkeypatch.setattr(wiz, "questionary", fake)
    # Skip the slow torch model load -- inject a synthetic introspection result.
    monkeypatch.setattr(
        wiz,
        "_introspect",
        lambda *a, **k: IntrospectionForm(
            promotable=["E", "nu"], outputs=["stress"], inputs=["strain"]
        ),
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(wiz, "run_compile", lambda argv: captured.update(argv=argv) or 0)

    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0
    argv = captured["argv"]
    assert argv[:3] == [str(_FIXTURE), "--model", "elasticity"]
    i = argv.index("--device")
    assert argv[i : i + 3] == ["--device", "cpu", "cuda"]
    assert argv[argv.index("-p") + 1] == "E"
    assert argv[argv.index("--dtype") + 1] == "float32"
    assert "--dynamic-batch" in argv
    assert "-d" not in argv  # derivatives declined


def test_wizard_example_shape_customize_per_variable(monkeypatch):
    """Decline uniform -> pick which inputs to customize -> prompt only those;
    non-blank answers become ``name=spec`` --example-batch-shape entries."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # target, name, derivatives?, rename?, devices, dtype, output, uniform?(No),
    # customize-which, strain=, temperature=, dynamic, review -> run.
    # (No "promote" prompt: promotable is empty.)
    fake = _FakeQuestionary(
        [
            "model",
            "m",
            False,
            False,  # rename any boundary variables?
            ["cpu"],
            "float64",
            "./aoti",
            False,  # uniform? -> customize
            ["strain", "temperature"],  # which inputs to customize
            "(2;100)",  # strain
            "",  # temperature (blank -> omitted)
            True,
            "run",
        ]
    )
    monkeypatch.setattr(wiz, "questionary", fake)
    monkeypatch.setattr(
        wiz,
        "_introspect",
        lambda *a, **k: IntrospectionForm(
            promotable=[],
            outputs=["stress"],
            inputs=["strain", "temperature"],
            input_types={"strain": "SR2", "temperature": "Scalar"},
        ),
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(wiz, "run_compile", lambda argv: captured.update(argv=argv) or 0)

    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0
    ebs = [
        captured["argv"][k + 1]
        for k, t in enumerate(captured["argv"])
        if t == "--example-batch-shape"
    ]
    assert ebs == ["strain=(2;100)"]  # temperature left blank -> omitted


def test_wizard_example_shape_uniform(monkeypatch):
    """Uniform -> one bare entry applied to all inputs."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # target, name, derivatives?, rename?, devices, dtype, output, uniform?, value,
    # dynamic, run. (No promote prompt: promotable is empty.)
    fake = _FakeQuestionary(
        ["model", "m", False, False, ["cpu"], "float64", "./aoti", True, "(4,)", True, "run"]
    )
    monkeypatch.setattr(wiz, "questionary", fake)
    monkeypatch.setattr(
        wiz,
        "_introspect",
        lambda *a, **k: IntrospectionForm(promotable=[], outputs=["stress"], inputs=["strain"]),
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(wiz, "run_compile", lambda argv: captured.update(argv=argv) or 0)

    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0
    argv = captured["argv"]
    assert argv[argv.index("--example-batch-shape") + 1] == "(4,)"


def test_wizard_derivatives_cross_product(monkeypatch):
    """The two toggle lists yield the cross-product of selected outputs x inputs
    (a promoted parameter is offered as a w.r.t. input)."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # target, name, promote, derivatives? -> yes, outputs, inputs, round-menu ->
    # done, rename? -> no, devices, dtype, output, uniform?(yes), uniform "",
    # dynamic, review -> run.
    fake = _FakeQuestionary(
        [
            "model",
            "m",
            ["E"],  # promote E
            True,  # compile derivatives?
            ["stress"],  # outputs
            ["strain", "E"],  # w.r.t. inputs (incl. promoted param)
            "done",  # round menu -> finish
            False,  # rename any boundary variables?
            ["cpu"],
            "float64",
            "./aoti",
            True,
            "",
            True,
            "run",
        ]
    )
    monkeypatch.setattr(wiz, "questionary", fake)
    monkeypatch.setattr(
        wiz,
        "_introspect",
        lambda *a, **k: IntrospectionForm(promotable=["E"], outputs=["stress"], inputs=["strain"]),
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(wiz, "run_compile", lambda argv: captured.update(argv=argv) or 0)

    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0
    argv = captured["argv"]
    dpairs = [argv[k + 1] for k, t in enumerate(argv) if t == "-d"]
    assert dpairs == ["stress:strain", "stress:E"]


def test_wizard_collects_renames(monkeypatch):
    """Accepting the rename step and typing new names for a chosen input / output /
    promoted parameter yields the three --rename-* flags."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # target, name, promote, derivatives?(no), rename?(yes),
    #   rename-inputs [strain] -> "eps",
    #   rename-outputs [stress] -> "sig",
    #   rename-params [E] -> "youngs",
    # devices, dtype, output, uniform?(yes), value, dynamic, run.
    fake = _FakeQuestionary(
        [
            "model",
            "m",
            ["E"],  # promote E (so it can be renamed)
            False,  # derivatives?
            True,  # rename any boundary variables?
            ["strain"],  # rename which inputs
            "eps",  # strain ->
            ["stress"],  # rename which outputs
            "sig",  # stress ->
            ["E"],  # rename which promoted parameters
            "youngs",  # E ->
            ["cpu"],
            "float64",
            "./aoti",
            True,
            "",
            True,
            "run",
        ]
    )
    monkeypatch.setattr(wiz, "questionary", fake)
    monkeypatch.setattr(
        wiz,
        "_introspect",
        lambda *a, **k: IntrospectionForm(promotable=["E"], outputs=["stress"], inputs=["strain"]),
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(wiz, "run_compile", lambda argv: captured.update(argv=argv) or 0)

    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0
    argv = captured["argv"]
    assert argv[argv.index("--rename-input") + 1] == "strain:eps"
    assert argv[argv.index("--rename-output") + 1] == "stress:sig"
    assert argv[argv.index("--rename-parameter") + 1] == "E:youngs"


def test_ask_renames_decline_and_cancel(monkeypatch):
    """_ask_renames: declining the gate returns empty maps; cancelling returns None;
    an unavailable-introspection call leaves the current selection unchanged."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    intro = IntrospectionForm(promotable=["E"], outputs=["stress"], inputs=["strain"])
    # Decline the gate -> all-empty maps.
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([False]))
    assert wiz._ask_renames(intro, ("E",), {}) == {"inputs": {}, "outputs": {}, "parameters": {}}
    # Cancel the gate -> None.
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([None]))
    assert wiz._ask_renames(intro, ("E",), {}) is None
    # No introspection -> keep current unchanged (no prompt consumed).
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([]))
    current = {"inputs": {"strain": "eps"}, "outputs": {}, "parameters": {}}
    assert wiz._ask_renames(None, (), current) == current


def test_wizard_rename_skipped_for_driver_target(monkeypatch):
    """The rename field is a no-op (no prompt) when the target is a driver."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # An empty answer script would raise if the branch tried to prompt.
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([]))
    state = wiz._default_state()
    state["target"] = "driver"
    state["name"] = "driver"
    state["rename"] = {"inputs": {"strain": "eps"}, "outputs": {}, "parameters": {}}
    assert wiz._ask_field("rename", str(_FIXTURE), state, {}) is True
    assert state["rename"] == {"inputs": {}, "outputs": {}, "parameters": {}}


def test_ask_derivatives_edit_add_clear(monkeypatch):
    """With existing selections the round menu runs: 'edit' toggles a flat list,
    'add' unions a new cross-product (hiding already-selected pairs), 'clear'
    empties. Each path ends with 'done'."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    intro = IntrospectionForm(promotable=[], outputs=["stress"], inputs=["strain", "temperature"])
    existing = ("stress:strain", "stress:temperature")

    # edit: menu -> "edit", flat checkbox keeps only stress:strain, menu -> "done".
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(["edit", ["stress:strain"], "done"]))
    assert wiz._ask_derivatives(intro, (), existing) == ["stress:strain"]

    # add: menu -> "add", outputs [stress], inputs [temperature] (strain hidden:
    # stress:strain already exists), unioned with existing, menu -> "done".
    monkeypatch.setattr(
        wiz, "questionary", _FakeQuestionary(["add", ["stress"], ["temperature"], "done"])
    )
    assert wiz._ask_derivatives(intro, (), ("stress:strain",)) == [
        "stress:strain",
        "stress:temperature",
    ]

    # clear: menu -> "clear", menu -> "done".
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(["clear", "done"]))
    assert wiz._ask_derivatives(intro, (), existing) == []


def test_wizard_edit_back_does_not_crash(monkeypatch):
    """Selecting the edit menu's '↩ back' returns to the review cleanly (regression:
    a None-valued Choice arrives as its title, so a distinct sentinel is used)."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # ... linear (deriv no, rename no, uniform example) ..., review -> edit,
    # edit-field -> back, review -> run.
    fake = _FakeQuestionary(
        [
            "model",
            "m",
            False,  # derivatives?
            False,  # rename?
            ["cpu"],
            "float64",
            "./aoti",
            True,
            "",
            True,
            "edit",
            wiz._BACK,
            "run",
        ]
    )
    monkeypatch.setattr(wiz, "questionary", fake)
    monkeypatch.setattr(
        wiz,
        "_introspect",
        lambda *a, **k: IntrospectionForm(promotable=[], outputs=["stress"], inputs=["strain"]),
    )
    ran: dict[str, bool] = {}
    monkeypatch.setattr(wiz, "run_compile", lambda argv: ran.update(ok=True) or 0)

    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0
    assert ran.get("ok") is True


# --------------------------------------------------------------------------- #
# neml2-compile --interactive argument handling                               #
# --------------------------------------------------------------------------- #


def test_compile_interactive_keeps_input_required_but_target_optional():
    """`-i` still requires INPUT.i (so shell path completion applies) but lets the
    model/driver be omitted (the wizard asks)."""
    from neml2.cli.aoti_compile import _build_arg_parser  # noqa: PLC2701, PLC0415

    args, _ = _build_arg_parser().parse_known_args(["-i", "in.i"])
    assert args.interactive is True
    assert str(args.input) == "in.i"
    assert args.model is None and args.driver is None
    # `-i` with no input file is an error (argparse-required positional).
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_known_args(["-i"])


def test_compile_noninteractive_requires_input_and_target():
    """Without `-i`, omitting input or model/driver must still error (SystemExit),
    preserving the original required behavior."""
    from neml2.cli.aoti_compile import main  # noqa: PLC0415

    with pytest.raises(SystemExit):
        main([])  # no input
    with pytest.raises(SystemExit):
        main([str(_FIXTURE)])  # input but no --model/--driver


def test_compile_interactive_delegates_to_wizard(monkeypatch):
    """`neml2-compile -i input.i` hands the input file off to the wizard."""
    pytest.importorskip("questionary")
    from neml2.cli import aoti_compile  # noqa: PLC0415

    captured: dict[str, object] = {}

    def _fake_wizard(*, input_file, initial_load):
        captured["input_file"] = input_file
        captured["initial_load"] = initial_load
        return 0

    # The wizard module is imported lazily inside main(); patch its symbol.
    import neml2.cli._compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "run_wizard", _fake_wizard)

    assert aoti_compile.main(["-i", str(_FIXTURE), "--load", "ext.py"]) == 0
    assert captured == {"input_file": str(_FIXTURE), "initial_load": ("ext.py",)}


def test_compile_interactive_missing_questionary_hint(monkeypatch, capsys):
    """`-i` with questionary absent prints an install hint and returns 1."""
    import builtins  # noqa: PLC0415

    from neml2.cli import aoti_compile  # noqa: PLC0415

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "questionary" or name.startswith("questionary."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert aoti_compile.main(["-i", str(_FIXTURE)]) == 1
    assert "pip install questionary" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Wizard coverage: early exits, cancels, review/edit loop, helpers            #
# --------------------------------------------------------------------------- #

# A model with a promotable param + one output/input, so the linear prompt
# sequence is deterministic for the scripted flows below.
_WIZ_INTRO = IntrospectionForm(promotable=["E"], outputs=["stress"], inputs=["strain"])
# Linear-pass answers (one per prompt) for that intro, with a uniform example
# shape: target, name, promote, derivatives?, rename?, devices, dtype, output,
# uniform?, uniform value, dynamic batch. (The rename step in the decline path is
# a single confirm answered False.)
_LINEAR_OK = [
    "model",
    "elasticity",
    ["E"],
    False,
    False,
    ["cpu"],
    "float64",
    "./aoti",
    True,
    "",
    True,
]


def _patch_intro(monkeypatch, wiz, intro=_WIZ_INTRO):
    monkeypatch.setattr(wiz, "_introspect", lambda *a, **k: intro)


def test_run_wizard_bad_input_file(monkeypatch, tmp_path):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([]))
    assert wiz.run_wizard(input_file=str(tmp_path / "missing.i")) == 1


def test_run_wizard_extension_load_error(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([]))
    rc = wiz.run_wizard(input_file=str(_FIXTURE), initial_load=("nonexistent_module_zzz_999",))
    assert rc == 1


@pytest.mark.parametrize("cancel_at", range(len(_LINEAR_OK)))
def test_run_wizard_cancel_at_any_prompt_aborts(monkeypatch, cancel_at):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(_LINEAR_OK[:cancel_at] + [None]))
    _patch_intro(monkeypatch, wiz)
    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 1


def test_run_wizard_name_empty_aborts(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(["model", "  "]))
    _patch_intro(monkeypatch, wiz)
    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 1


def test_run_wizard_review_quit(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([*_LINEAR_OK, "quit"]))
    _patch_intro(monkeypatch, wiz)
    monkeypatch.setattr(wiz, "run_compile", lambda argv: 0)
    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0


def test_run_wizard_review_cancel_aborts(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([*_LINEAR_OK, None]))
    _patch_intro(monkeypatch, wiz)
    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 1


def test_run_wizard_review_validate_problem_then_quit(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # devices = [] -> validate_state flags it -> "run" loops back -> "quit".
    # (deriv no, rename no, then devices [].)
    answers = ["model", "elasticity", ["E"], False, False, [], "float64", "./aoti", True, "", True]
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([*answers, "run", "quit"]))
    _patch_intro(monkeypatch, wiz)
    monkeypatch.setattr(wiz, "run_compile", lambda argv: 0)  # must NOT be called
    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0


def test_run_wizard_review_edit_target_reasks_name(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    # review -> edit -> field "target" -> target "model" -> name re-ask "elasticity" -> run.
    answers = [*_LINEAR_OK, "edit", "target", "model", "elasticity", "run"]
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(answers))
    _patch_intro(monkeypatch, wiz)
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(wiz, "run_compile", lambda argv: captured.update(argv=argv) or 0)
    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0
    assert captured["argv"][:3] == [str(_FIXTURE), "--model", "elasticity"]


def test_run_wizard_introspection_failure_fallback(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    def _boom(*a, **k):
        raise RuntimeError("cannot load")

    monkeypatch.setattr(wiz, "_introspect", _boom)
    # No promote/derivative prompts (intro is None); example uses the free-text
    # fallback. target, name, devices, dtype, output, example-fallback, dynamic, quit.
    answers = ["model", "m", ["cpu"], "float64", "./aoti", "(2,)", True, "quit"]
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(answers))
    assert wiz.run_wizard(input_file=str(_FIXTURE)) == 0


def test_ask_example_shape_cancel_paths(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    _patch_intro(monkeypatch, wiz, IntrospectionForm(outputs=["stress"], inputs=["strain"]))

    def _state():
        s = wiz._default_state()
        s["name"] = "m"
        return s

    # Decline uniform, then cancel the "which inputs" checkbox.
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([False, None]))
    assert wiz._ask_example_shape(str(_FIXTURE), _state(), {}) is False
    # Decline uniform, pick strain, then cancel its per-variable text.
    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([False, ["strain"], None]))
    assert wiz._ask_example_shape(str(_FIXTURE), _state(), {}) is False


def test_bulk_select_pairs_paths(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    def q(answers):
        monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(answers))

    # avail_out empty -> nothing left to add.
    q([])
    assert wiz._bulk_select_pairs(["a"], ["x"], ["a:x"]) == []
    # outputs cancelled / empty.
    q([None])
    assert wiz._bulk_select_pairs(["a"], ["x"], []) is None
    q([[]])
    assert wiz._bulk_select_pairs(["a"], ["x"], []) == []
    # avail_in empty (the one selected output is already fully paired).
    q([["a"]])
    assert wiz._bulk_select_pairs(["a", "b"], ["x"], ["a:x"]) == []
    # inputs cancelled / empty.
    q([["a"], None])
    assert wiz._bulk_select_pairs(["a"], ["x", "y"], []) is None
    q([["a"], []])
    assert wiz._bulk_select_pairs(["a"], ["x", "y"], []) == []
    # normal: cross-product, excluding the already-selected pair.
    q([["a"], ["x", "y"]])
    assert wiz._bulk_select_pairs(["a"], ["x", "y"], ["a:x"]) == ["a:y"]


def test_run_compile_invokes_subprocess(monkeypatch):
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([]))
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(wiz.subprocess, "call", lambda cmd: captured.update(cmd=cmd) or 0)
    assert wiz.run_compile(["in.i", "--model", "m"]) == 0
    assert captured["cmd"][1:3] == ["-m", "neml2.cli.aoti_compile"]
    assert captured["cmd"][-3:] == ["in.i", "--model", "m"]

    def _interrupt(cmd):
        raise KeyboardInterrupt

    monkeypatch.setattr(wiz.subprocess, "call", _interrupt)
    assert wiz.run_compile(["x"]) == 130


def test_introspect_real_model_and_driver():
    """Exercise the real introspection body (model + driver-resolution branch)."""
    pytest.importorskip("questionary")
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    by_model = wiz._introspect(str(_FIXTURE), "model", "elasticity")
    assert by_model.outputs and by_model.inputs
    by_driver = wiz._introspect(str(_FIXTURE), "driver", "driver")
    assert by_driver.outputs == by_model.outputs  # driver resolves to the same model


def test_argv_builder_empty_optionals():
    """All-empty optionals exercise the skip branches in build_compile_argv."""
    argv = build_compile_argv(
        FormState(input_file="", name="", output_dir="", devices=(), dtype="")
    )
    assert argv == []


def test_ask_jobs_prompts_on_multi_segment(monkeypatch):
    """A multi-segment model shows the segment plan and prompts for a process
    count, consuming exactly one scripted answer."""
    pytest.importorskip("questionary")
    import neml2  # noqa: F401, PLC0415 — registers models
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(["3"]))
    state = wiz._default_state()
    state["target"] = "model"
    state["name"] = "model"
    assert wiz._ask_field("jobs", str(_COMPOSED_I), state, {}) is True
    assert state["jobs"] == 3


def test_ask_jobs_single_segment_no_prompt(monkeypatch):
    """A single-segment model pins jobs=1 without prompting -- the empty script
    would raise StopIteration if a prompt were shown."""
    pytest.importorskip("questionary")
    import neml2  # noqa: F401, PLC0415 — registers models
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary([]))
    state = wiz._default_state()
    state["target"] = "model"
    state["name"] = "model"
    assert wiz._ask_field("jobs", str(_ELASTICITY_I), state, {}) is True
    assert state["jobs"] == 1


def test_ask_jobs_clamps_to_segment_count(monkeypatch):
    """On one device, requesting more processes than segments is clamped to the
    segment count (the composed fixture has 3 segments)."""
    pytest.importorskip("questionary")
    import neml2  # noqa: F401, PLC0415 — registers models
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(["99"]))
    state = wiz._default_state()
    state["target"] = "model"
    state["name"] = "model"
    assert wiz._ask_field("jobs", str(_COMPOSED_I), state, {}) is True
    assert state["jobs"] == 3


def test_ask_jobs_clamps_to_grid_over_multiple_devices(monkeypatch):
    """The clamp is the grid size #devices × #segments: 2 devices × 3 segments = 6
    (computed from the chosen device list; no CUDA needed to plan)."""
    pytest.importorskip("questionary")
    import neml2  # noqa: F401, PLC0415 — registers models
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(["99"]))
    state = wiz._default_state()
    state["target"] = "model"
    state["name"] = "model"
    state["devices"] = ("cpu", "cuda")
    assert wiz._ask_field("jobs", str(_COMPOSED_I), state, {}) is True
    assert state["jobs"] == 6


def test_ask_jobs_single_segment_multi_device_prompts(monkeypatch):
    """A single-segment model on TWO devices is still a 2-cell grid, so parallel
    compilation applies and the wizard prompts (rather than pinning to 1)."""
    pytest.importorskip("questionary")
    import neml2  # noqa: F401, PLC0415 — registers models
    from neml2.cli import _compile_wizard as wiz  # noqa: PLC2701, PLC0415

    monkeypatch.setattr(wiz, "questionary", _FakeQuestionary(["2"]))
    state = wiz._default_state()
    state["target"] = "model"
    state["name"] = "model"
    state["devices"] = ("cpu", "cuda")
    assert wiz._ask_field("jobs", str(_ELASTICITY_I), state, {}) is True
    assert state["jobs"] == 2
