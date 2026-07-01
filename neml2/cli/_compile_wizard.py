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

"""The interactive Questionary flow behind ``neml2-compile --interactive``.

Imported lazily by :func:`neml2.cli.aoti_compile.main` only after the
``questionary`` dependency is confirmed present, so ``neml2-compile``'s normal
(non-interactive) path never imports it. The framework-free core it builds on
lives in :mod:`neml2.cli._compile_interactive`.

The flow has two parts: a guided linear pass that asks each option in turn
(target, name, parameters, derivatives, devices, dtype, output dir, batch
shape), then a **review screen** that shows every answer plus the equivalent
``neml2-compile`` command and lets the user jump back to *any* field to change it
before running. Questionary has no built-in "go back", so this review/edit loop
is how earlier answers are revisited. On confirmation it runs the *exact*
previewed command as a subprocess that inherits this terminal, so the tool itself
does no log capture -- ``neml2-compile``'s own output streams straight to the
user.

Every prompt returns ``None`` when the user cancels (Ctrl-C / Esc); during the
linear pass that aborts the wizard, while in the review/edit loop it just returns
to the review.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import questionary

from ._compile_interactive import (
    FormState,
    IntrospectionForm,
    build_compile_argv,
    introspection_to_form,
    list_section_entries,
    preview_command,
    validate_state,
)

# (state key, human label) for every editable field, in the order the linear
# pass asks them and the order the edit menu lists them.
_FIELDS: list[tuple[str, str]] = [
    ("target", "Target"),
    ("name", "Name"),
    ("promoted", "Parameters"),
    ("derivatives", "Derivatives"),
    ("rename", "Renames"),
    ("devices", "Devices"),
    ("dtype", "dtype"),
    ("output_dir", "Output dir"),
    ("example_shape", "Example batch shape"),
    ("dynamic_batch", "Dynamic batch"),
]

# Distinct sentinel for the edit menu's "back" choice. A questionary ``Choice``
# substitutes the *title* for a ``None`` value, so ``None`` cannot be used to
# mean "no field" -- it would arrive as the string "↩ back".
_BACK = object()

# One-line explanation per field, printed as a muted line just above its prompt.
# (Questionary's high-level API has no persistent bottom toolbar, and its
# ``instruction=`` argument is not uniformly usable -- ``path`` rejects it and
# ``confirm`` lets it clobber the ``[y/N]`` hint -- so help is printed instead.)
_HELP: dict[str, str] = {
    "target": "Compile a [Models] block directly, or the model a [Drivers] block runs.",
    "name": "The block to compile (its type is shown in parentheses).",
    "promoted": "Checked params become runtime-settable graph inputs instead of baked "
    "constants; space toggles, enter confirms.",
    "derivatives": "Also compile Jacobian/JVP graphs for chosen output:input pairs "
    "(needed for jvp()/jacobian() at runtime).",
    "rename": "Rename input/output/parameter names at the compiled boundary for a "
    "downstream consumer; the model internals keep the original names. Model target only.",
    "devices": "Target device(s), baked at export; one artifact is emitted per device.",
    "dtype": "Floating-point precision baked into the artifact.",
    "output_dir": "Directory for the .pt2 artifacts and the runnable HIT stub.",
    "example_shape": "Trace-time shape per input; '(dyn;sub)' splits dynamic from "
    "sub-batch axes, e.g. (2;100). Blank = default.",
    "dynamic_batch": "Let the leading batch dimension vary at runtime (vs. pinned to "
    "the example shape).",
}


def run_wizard(*, input_file: str, initial_load: tuple[str, ...] = ()) -> int:
    """Drive the interactive compile wizard for *input_file*. Returns an exit code.

    *input_file* is taken from the ``neml2-compile`` command line (a required
    positional) rather than prompted for -- the shell's own path completion beats
    an in-prompt one, and an input file is needed to begin with.
    """
    input_file = input_file.strip()
    if not input_file or not Path(input_file).expanduser().exists():
        print(f"Error: input file not found: {input_file!r}", file=sys.stderr)
        return 1
    questionary.print(f"Input file: {input_file}", style="bold")

    # User extensions must register before we list / load anything.
    if initial_load:
        from ._extensions import load_user_extensions  # noqa: PLC0415

        try:
            load_user_extensions(list(initial_load))
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    state = _default_state()
    cache: dict[tuple[str, str], IntrospectionForm | None] = {}

    # Guided linear pass -- cancelling any prompt aborts the wizard.
    for field, _label in _FIELDS:
        if not _ask_field(field, input_file, state, cache):
            return _aborted()

    # Review / edit loop -- the "go back to a previous question" surface.
    while True:
        fs = _form_state(input_file, state, initial_load)
        _print_review(fs)
        action = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Run the compile", "run"),
                questionary.Choice("Edit a field", "edit"),
                questionary.Choice("Quit without compiling", "quit"),
            ],
        ).ask()
        if action is None:
            return _aborted()
        if action == "quit":
            questionary.print("Quit. Copy the command above to run it later.", style="italic")
            return 0
        if action == "run":
            problems = validate_state(fs)
            if problems:
                for problem in problems:
                    questionary.print(f"  - {problem}", style="fg:ansired")
                continue
            return run_compile(build_compile_argv(fs))
        # action == "edit": pick a field and re-ask just that one.
        field = questionary.select(
            "Edit which field?",
            choices=[questionary.Choice(label, key) for key, label in _FIELDS]
            + [questionary.Choice("↩ back", _BACK)],
        ).ask()
        if field is None or field is _BACK:
            continue
        _ask_field(field, input_file, state, cache)
        # Changing the target invalidates the name (different section), so re-pick.
        if field == "target":
            _ask_field("name", input_file, state, cache)


def _empty_renames() -> dict[str, dict[str, str]]:
    return {"inputs": {}, "outputs": {}, "parameters": {}}


def _default_state() -> dict:
    return {
        "target": "model",
        "name": "",
        "promoted": (),
        "derivatives": (),
        # {"inputs"|"outputs"|"parameters": {orig: new}} -- shallow boundary renames.
        "rename": _empty_renames(),
        "devices": ("cpu",),
        "dtype": "float64",
        "output_dir": "./aoti",
        "example_shape": (),  # tuple of raw `--example-batch-shape` entries
        "dynamic_batch": True,
    }


def _ask_field(field: str, input_file: str, state: dict, cache: dict) -> bool:
    """Prompt for one field, updating *state*. Returns False if the user cancels."""
    # Show the field's explanation as a muted line above the prompt. (Questionary's
    # `instruction=` is not uniformly safe -- `path` rejects it and `confirm`
    # replaces its [y/N] hint with it -- so help goes here instead.)
    help_text = _HELP.get(field)
    if help_text:
        questionary.print(help_text, style="italic fg:ansibrightblack")

    if field == "target":
        v = questionary.select(
            "Target:", choices=["model", "driver"], default=state["target"]
        ).ask()
        if v is None:
            return False
        if v != state["target"]:
            state["target"] = v
            state["name"] = ""  # name belongs to the old section
            _reset_model_dependent(state)
        return True

    if field == "name":
        section = "Models" if state["target"] == "model" else "Drivers"
        entries = list_section_entries(input_file, section)
        label = f"{state['target'].capitalize()} name:"
        if entries:
            choices = [questionary.Choice(f"{n}  ({t})" if t else n, n) for n, t in entries]
            default = state["name"] if any(n == state["name"] for n, _ in entries) else None
            v = questionary.select(label, choices=choices, default=default).ask()
        else:
            v = questionary.text(label, default=state["name"]).ask()
        if v is None:
            return False
        v = v.strip()
        if not v:
            questionary.print("A name is required.", style="fg:ansired")
            return False
        if v != state["name"]:
            state["name"] = v
            _reset_model_dependent(state)
        _ensure_intro(input_file, state, cache)
        return True

    if field == "promoted":
        intro = _ensure_intro(input_file, state, cache)
        if not intro or not intro.promotable:
            state["promoted"] = ()
            questionary.print("(model has no promotable parameters)", style="italic")
            return True
        v = questionary.checkbox(
            "Promote parameters to runtime inputs:",
            choices=[
                questionary.Choice(p, p, checked=(p in state["promoted"])) for p in intro.promotable
            ],
        ).ask()
        if v is None:
            return False
        state["promoted"] = tuple(v)
        return True

    if field == "derivatives":
        intro = _ensure_intro(input_file, state, cache)
        specs = _ask_derivatives(intro, tuple(state["promoted"]), tuple(state["derivatives"]))
        if specs is None:
            return False
        state["derivatives"] = tuple(specs)
        return True

    if field == "rename":
        # Renaming is a model-target-only boundary feature (the driver stub keeps
        # the original [Drivers] wiring). Skip cleanly in driver mode.
        if state["target"] != "model":
            state["rename"] = _empty_renames()
            questionary.print(
                "(renaming is only available with a model target, not a driver)",
                style="italic",
            )
            return True
        intro = _ensure_intro(input_file, state, cache)
        renames = _ask_renames(intro, tuple(state["promoted"]), state.get("rename", {}))
        if renames is None:
            return False
        state["rename"] = renames
        return True

    if field == "devices":
        v = questionary.checkbox(
            "Devices:",
            choices=[
                questionary.Choice("cpu", "cpu", checked="cpu" in state["devices"]),
                questionary.Choice("cuda", "cuda", checked="cuda" in state["devices"]),
            ],
        ).ask()
        if v is None:
            return False
        state["devices"] = tuple(v)
        return True

    if field == "dtype":
        v = questionary.select(
            "dtype:", choices=["float64", "float32"], default=state["dtype"]
        ).ask()
        if v is None:
            return False
        state["dtype"] = v
        return True

    if field == "output_dir":
        v = questionary.path(
            "Output dir:", default=state["output_dir"], only_directories=True
        ).ask()
        if v is None:
            return False
        state["output_dir"] = v.strip()
        return True

    if field == "example_shape":
        return _ask_example_shape(input_file, state, cache)

    if field == "dynamic_batch":
        v = questionary.confirm("Dynamic batch?", default=bool(state["dynamic_batch"])).ask()
        if v is None:
            return False
        state["dynamic_batch"] = bool(v)
        return True

    raise ValueError(f"unknown field {field!r}")  # pragma: no cover -- defensive


def _ask_example_shape(input_file: str, state: dict, cache: dict) -> bool:
    """Ask the example batch shape: a uniform shape for all inputs, or a custom
    shape for a chosen subset of inputs.

    First ask whether to apply one uniform shape; if so, prompt that single value.
    Otherwise let the user pick which inputs to customize and prompt only those.
    Non-blank per-input answers become ``name=spec`` ``--example-batch-shape``
    entries; a non-blank uniform answer is a single bare entry. When introspection
    is unavailable, fall back to one free-text prompt.
    """
    intro = _ensure_intro(input_file, state, cache)
    entries = tuple(state["example_shape"])
    # Split existing entries into a uniform value vs. per-variable specs (defaults).
    current_per_var = {
        k.strip(): v.strip()
        for entry in entries
        if "=" in entry and not entry.lstrip().startswith("(")
        for k, _, v in [entry.partition("=")]
    }
    current_uniform = entries[0] if len(entries) == 1 and "=" not in entries[0] else ""

    if not intro or not intro.inputs:
        prev = entries[0] if entries else ""
        v = questionary.text(
            "Example batch shape (optional, e.g. (2,) or strain=(2;100)):", default=prev
        ).ask()
        if v is None:
            return False
        v = v.strip()
        state["example_shape"] = (v,) if v else ()
        return True

    uniform = questionary.confirm(
        "Apply a uniform example batch shape to all inputs?", default=bool(current_uniform)
    ).ask()
    if uniform is None:
        return False
    if uniform:
        v = questionary.text(
            "Uniform example batch shape (e.g. (2,) or (2;100); blank = default):",
            default=current_uniform,
        ).ask()
        if v is None:
            return False
        v = v.strip()
        state["example_shape"] = (v,) if v else ()
        return True

    # Customize: pick which inputs to set, then prompt only those.
    which = questionary.checkbox(
        "Customize the example shape for which inputs? (none = use defaults)",
        choices=[
            questionary.Choice(
                f"{n} ({intro.input_types.get(n) or '?'})", n, checked=(n in current_per_var)
            )
            for n in intro.inputs
        ],
    ).ask()
    if which is None:
        return False
    specs: dict[str, str] = {}
    for name in which:
        typ = intro.input_types.get(name) or "?"
        v = questionary.text(f"  {name} ({typ}):", default=current_per_var.get(name, "")).ask()
        if v is None:
            return False
        v = v.strip()
        if v:
            specs[name] = v
    state["example_shape"] = tuple(f"{k}={v}" for k, v in specs.items())
    return True


def _reset_model_dependent(state: dict) -> None:
    """Drop parameter / derivative / rename selections that belong to the previous model."""
    renames = state.get("rename", {})
    if state["promoted"] or state["derivatives"] or any(renames.values()):
        questionary.print(
            "(cleared parameter / derivative / rename selections for the new model)",
            style="italic",
        )
    state["promoted"] = ()
    state["derivatives"] = ()
    state["rename"] = _empty_renames()


def _ensure_intro(input_file: str, state: dict, cache: dict) -> IntrospectionForm | None:
    """Introspect the current (target, name), cached. ``None`` if it can't load."""
    name = state["name"]
    if not name:
        return None
    key = (state["target"], name)
    if key in cache:
        return cache[key]
    try:
        questionary.print(f"Loading model {name!r} ...", style="italic")
        intro: IntrospectionForm | None = _introspect(input_file, state["target"], name)
    except Exception as exc:  # noqa: BLE001
        questionary.print(f"Could not introspect model: {exc}", style="fg:ansired")
        intro = None
    cache[key] = intro
    return intro


def _form_state(input_file: str, state: dict, initial_load: tuple[str, ...]) -> FormState:
    ren = state.get("rename", {})

    def _specs(ns: str) -> tuple[str, ...]:
        return tuple(f"{orig}:{new}" for orig, new in ren.get(ns, {}).items())

    return FormState(
        input_file=input_file,
        target=state["target"],
        name=state["name"],
        devices=tuple(state["devices"]),
        dtype=state["dtype"],
        output_dir=state["output_dir"],
        promoted=tuple(state["promoted"]),
        derivatives=tuple(state["derivatives"]),
        rename_inputs=_specs("inputs"),
        rename_outputs=_specs("outputs"),
        rename_parameters=_specs("parameters"),
        example_shape=tuple(state["example_shape"]),
        dynamic_batch=state["dynamic_batch"],
        load=tuple(initial_load),
    )


def _print_review(fs: FormState) -> None:
    questionary.print("\nReview:", style="bold")
    ren_bits = [
        f"{prefix} {spec.replace(':', '->')}"
        for prefix, specs in (
            ("in", fs.rename_inputs),
            ("out", fs.rename_outputs),
            ("param", fs.rename_parameters),
        )
        for spec in specs
    ]
    rows = [
        ("Target", f"{fs.target}: {fs.name or '(unset)'}"),
        ("Parameters", ", ".join(fs.promoted) or "(none)"),
        ("Derivatives", ", ".join(fs.derivatives) or "(none)"),
        ("Renames", ", ".join(ren_bits) or "(none)"),
        ("Devices", ", ".join(fs.devices) or "(none)"),
        ("dtype", fs.dtype),
        ("Output dir", fs.output_dir),
        ("Example shape", ", ".join(fs.example_shape) or "(none)"),
        ("Dynamic batch", "yes" if fs.dynamic_batch else "no"),
    ]
    for label, value in rows:
        questionary.print(f"  {label:<14} {value}")
    questionary.print("\nCommand:", style="bold")
    questionary.print("  " + preview_command(fs))


def _aborted() -> int:
    print("Aborted.", file=sys.stderr)
    return 1


def _introspect(input_file: str, target: str, name: str) -> IntrospectionForm:
    """Load + introspect the chosen model (resolving a driver to its model)."""
    from ..factory import load_input  # noqa: PLC0415
    from .aoti_compile import driver_target_model  # noqa: PLC0415
    from .inspect import _model_to_dict  # noqa: PLC2701, PLC0415

    model_name = driver_target_model(Path(input_file), name) if target == "driver" else name
    model = load_input(input_file).get_model(model_name)
    return introspection_to_form(_model_to_dict(model))


def _ask_derivatives(
    intro: IntrospectionForm | None, promoted: tuple[str, ...], current: tuple[str, ...]
) -> list[str] | None:
    """Select which ``OUT:IN`` derivative pairs to compile.

    Returns the new list of ``out:in`` specs (``[]`` = no derivatives) or ``None``
    if the user cancelled. With nothing selected yet it gates on a confirm then
    runs one bulk round (two toggle-able lists -- outputs, then w.r.t.
    inputs/params -- whose cross-product is the pair set). It then loops on a
    round menu: add another bulk selection (hiding pairs already chosen), edit the
    current pairs as a flat toggle list, clear, or finish.
    """
    if intro is None:
        return list(current)  # can't introspect -> leave the selection unchanged
    outputs = intro.outputs
    # Structural inputs plus any promoted parameter (parameter derivatives).
    inputs = list(intro.inputs) + [p for p in promoted if p not in intro.inputs]
    if not outputs or not inputs:
        return []

    pairs = list(current)
    if not pairs:
        # Gate the no-derivatives case, then run one initial bulk round.
        want = questionary.confirm("Compile derivative (Jacobian) graphs?", default=False).ask()
        if want is None:
            return None
        if not want:
            return []
        new = _bulk_select_pairs(outputs, inputs, pairs)
        if new is None:
            return None
        pairs = list(dict.fromkeys([*pairs, *new]))

    # Round menu -- add more / edit / clear / finish.
    while True:
        if pairs:
            questionary.print(f"Selected derivatives ({len(pairs)}):", style="bold")
            for pair in pairs:
                questionary.print(f"  - {pair}")
        else:
            questionary.print("No derivatives selected yet.", style="italic")
        action = questionary.select(
            "Derivatives -- what next?",
            choices=[
                questionary.Choice("Add a selection", "add"),
                questionary.Choice("Edit existing selections", "edit"),
                questionary.Choice("Clear all", "clear"),
                questionary.Choice("Done (continue)", "done"),
            ],
            default="done",
        ).ask()
        if action is None:
            return None
        if action == "done":
            return pairs
        if action == "clear":
            pairs = []
        elif action == "edit":
            kept = questionary.checkbox(
                "Keep which derivative pairs? (uncheck to remove)",
                choices=[questionary.Choice(p, p, checked=True) for p in pairs],
            ).ask()
            if kept is not None:
                pairs = list(kept)
        else:  # "add" -- new pairs are unioned with the (hidden) existing ones.
            new = _bulk_select_pairs(outputs, inputs, pairs)
            if new is not None:
                pairs = list(dict.fromkeys([*pairs, *new]))


def _ask_renames(
    intro: IntrospectionForm | None,
    promoted: tuple[str, ...],
    current: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]] | None:
    """Collect shallow boundary renames per namespace.

    Returns ``{"inputs"|"outputs"|"parameters": {orig: new}}`` (empty sub-maps =
    no renames) or ``None`` if the user cancelled. Gates on a single confirm, then
    for each namespace lets the user check which names to rename (candidates come
    from introspection: the model's inputs / outputs, and the currently-promoted
    parameters) and type each new boundary name. An empty or unchanged new name
    leaves that variable as-is.
    """
    if intro is None:
        return dict(current)  # can't introspect -> leave the selection unchanged
    namespaces = [
        ("inputs", "inputs", list(intro.inputs)),
        ("outputs", "outputs", list(intro.outputs)),
        ("parameters", "promoted parameters", list(promoted)),
    ]
    has_any = any(current.get(ns) for ns, _, _ in namespaces)
    want = questionary.confirm("Rename any boundary variables?", default=has_any).ask()
    if want is None:
        return None
    if not want:
        return _empty_renames()

    result = _empty_renames()
    for ns, label, candidates in namespaces:
        cur: dict[str, str] = current.get(ns, {})
        if not candidates:
            continue
        which = questionary.checkbox(
            f"Rename which {label}? (leave unchecked to keep the original name)",
            choices=[questionary.Choice(c, c, checked=(c in cur)) for c in candidates],
        ).ask()
        if which is None:
            return None
        mapping: dict[str, str] = {}
        for name in which:
            default_new = cur[name] if name in cur else name
            new = questionary.text(f"  {name} -> ", default=default_new).ask()
            if new is None:
                return None
            new = new.strip()
            if new and new != name:
                mapping[name] = new
        result[ns] = mapping
    return result


def _bulk_select_pairs(
    outputs: list[str], inputs: list[str], existing: list[str]
) -> list[str] | None:
    """Two toggle-able lists (outputs x w.r.t. inputs); return their cross-product
    as ``out:in`` specs, excluding any already in *existing*. ``None`` if
    cancelled, ``[]`` if a side is empty or everything is already selected.

    Items already fully covered by *existing* are hidden: an output paired with
    every input, then an input paired with every selected output.
    """
    taken = set(existing)
    avail_out = [o for o in outputs if any(f"{o}:{i}" not in taken for i in inputs)]
    if not avail_out:
        questionary.print("(all output:input pairs are already selected)", style="italic")
        return []
    outs = questionary.checkbox(
        "Differentiate which outputs?",
        choices=[questionary.Choice(o, o) for o in avail_out],
    ).ask()
    if outs is None:
        return None
    if not outs:
        return []
    avail_in = [i for i in inputs if any(f"{o}:{i}" not in taken for o in outs)]
    if not avail_in:
        questionary.print("(all pairs for those outputs are already selected)", style="italic")
        return []
    ins = questionary.checkbox(
        "With respect to which inputs / parameters?",
        choices=[questionary.Choice(i, i) for i in avail_in],
    ).ask()
    if ins is None:
        return None
    if not ins:
        return []
    return [f"{o}:{i}" for o in outs for i in ins if f"{o}:{i}" not in taken]


def run_compile(argv: list[str]) -> int:
    """Run ``neml2-compile`` with *argv*, inheriting stdio so its output shows live."""
    questionary.print(f"\n$ neml2-compile {' '.join(argv)}\n", style="bold")
    try:
        return subprocess.call([sys.executable, "-m", "neml2.cli.aoti_compile", *argv])
    except KeyboardInterrupt:
        return 130


__all__ = ["run_wizard", "run_compile"]
