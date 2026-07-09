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

"""Framework-free core for ``neml2-compile --interactive``.

The interactive (``-i`` / ``--interactive``) mode of ``neml2-compile`` walks the
user through the tool's wide, interdependent flag surface as a sequence of
prompts, introspecting the chosen model to offer its promotable parameters and
derivative variables. This module holds the **pure** pieces of that flow -- the
state dataclasses, the ``neml2-compile`` argv builder, and the introspection /
validation / block-listing helpers -- with no dependency on ``questionary``, so
they are unit-testable on their own. The interactive prompt loop that consumes
them lives in :mod:`neml2.cli._compile_wizard`.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FormState:
    """A snapshot of the wizard's answers, sufficient to build the compile argv.

    Pure data -- the wizard collects answers into one of these and every
    command-building / validation helper consumes it, so the non-interactive
    logic is testable in isolation.
    """

    input_file: str = ""
    target: str = "model"  # "model" | "driver"
    name: str = ""
    devices: tuple[str, ...] = ("cpu",)
    dtype: str = "float64"
    output_dir: str = "./aoti"
    promoted: tuple[str, ...] = ()
    derivatives: tuple[str, ...] = ()  # "OUT:IN" specs
    # Shallow boundary renames, each a tuple of "ORIG:NEW" specs (model mode only).
    rename_inputs: tuple[str, ...] = ()
    rename_outputs: tuple[str, ...] = ()
    rename_parameters: tuple[str, ...] = ()
    example_shape: tuple[str, ...] = ()  # raw --example-batch-shape entries
    dynamic_batch: bool | None = None  # None => emit neither flag
    jobs: int = 1  # parallel compile processes (-j); 1 == serial
    load: tuple[str, ...] = ()  # --load extension modules


@dataclass(frozen=True)
class PlanSummary:
    """The segment / artifact breakdown shown before choosing a process count.

    ``segments`` is one ``(index, kind, basename, artifacts)`` tuple per compiled
    segment; ``files`` is every file the compile will generate (each segment's
    ``.pt2`` graphs, the ``_meta.json``, and the ``_aoti.i`` stub). Computed by
    :func:`plan_summary` from the same planner the real compile uses, so the
    preview matches what is produced.
    """

    segments: list[tuple[int, str, str, tuple[str, ...]]] = field(default_factory=list)
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntrospectionForm:
    """The model facts the wizard needs, reduced from :func:`inspect._model_to_dict`.

    ``promotable`` are the dotted names ``-p`` accepts (parameters + buffers);
    ``outputs`` / ``inputs`` are the variable names that populate the derivative
    ``OUT`` / ``IN`` choices; ``input_types`` maps each input name to its tensor
    type (e.g. ``"SR2"``) for display in the example-batch-shape hint.
    """

    promotable: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    input_types: dict[str, str] = field(default_factory=dict)


def _dedupe(items: tuple[str, ...] | list[str]) -> list[str]:
    """Order-preserving de-duplication (mirrors ``neml2-compile``'s device handling)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_compile_argv(state: FormState) -> list[str]:
    """Build the ``neml2-compile`` argv (sans program name) from *state*.

    The argument order mirrors :func:`neml2.cli.aoti_compile._build_arg_parser`
    so the previewed command is exactly what the real CLI parses. The same argv
    drives both the preview (:func:`preview_command`) and the actual run -- a
    single builder guarantees "what you preview is what runs".
    """
    argv: list[str] = []

    # Positional input first -- keeps argparse from folding it into the
    # nargs="+" --device list below.
    if state.input_file:
        argv.append(state.input_file)

    # Exactly one of --model / --driver (the CLI's mutually-exclusive group).
    flag = "--driver" if state.target == "driver" else "--model"
    if state.name:
        argv += [flag, state.name]

    if state.output_dir:
        argv += ["--output-dir", state.output_dir]

    devices = _dedupe(state.devices)
    if devices:
        # Single nargs="+" flag: `--device cpu cuda`. Safe because every token
        # that follows starts with `-`, so argparse stops consuming here.
        argv += ["--device", *devices]

    if state.dtype:
        argv += ["--dtype", state.dtype]

    # Only emit -j when parallel; the default (serial, 1) is the CLI default, so
    # omitting it keeps the previewed command clean.
    if state.jobs > 1:
        argv += ["-j", str(state.jobs)]

    for name in state.promoted:
        argv += ["-p", name]

    for spec in state.derivatives:
        argv += ["-d", spec]

    # Shallow boundary renames (model mode only; the CLI rejects them with
    # --driver). Emitted per namespace, matching the three CLI flags.
    for spec in state.rename_inputs:
        argv += ["--rename-input", spec]
    for spec in state.rename_outputs:
        argv += ["--rename-output", spec]
    for spec in state.rename_parameters:
        argv += ["--rename-parameter", spec]

    for spec in state.example_shape:
        argv += ["--example-batch-shape", spec]

    if state.dynamic_batch is True:
        argv.append("--dynamic-batch")
    elif state.dynamic_batch is False:
        argv.append("--no-dynamic-batch")

    for path in state.load:
        argv += ["--load", path]

    return argv


def preview_command(state: FormState) -> str:
    """Render the shell command the wizard would run, with shell-safe quoting.

    ``shlex.join`` quotes specs containing ``;`` / ``(`` / spaces (e.g.
    ``strain=(2;100)``) so the previewed line is copy-pasteable into a shell.
    """
    return "neml2-compile " + shlex.join(build_compile_argv(state))


def introspection_to_form(data: dict[str, Any]) -> IntrospectionForm:
    """Reduce :func:`neml2.cli.inspect._model_to_dict` output to wizard choices.

    Promotable names are parameters followed by buffers (both are valid ``-p``
    targets, per :func:`neml2.cli.aoti_export._validate_promoted`); the
    derivative choices draw from the output / (structural) input variable names.
    """
    promotable = [p["name"] for p in data.get("parameters", [])]
    promotable += [b["name"] for b in data.get("buffers", [])]
    outputs = [o["name"] for o in data.get("outputs", [])]
    inputs = [i["name"] for i in data.get("inputs", [])]
    input_types = {i["name"]: i.get("type", "") for i in data.get("inputs", [])}
    return IntrospectionForm(
        promotable=promotable, outputs=outputs, inputs=inputs, input_types=input_types
    )


def validate_state(state: FormState) -> list[str]:
    """Return human-readable problems that block a compile (empty == ready)."""
    problems: list[str] = []
    if not state.input_file.strip():
        problems.append("Input file is required.")
    elif not Path(state.input_file).expanduser().exists():
        problems.append(f"Input file not found: {state.input_file}")
    if not state.name.strip():
        problems.append(f"A {state.target} name is required.")
    if not state.devices:
        problems.append("Select at least one device.")
    if state.jobs < 1:
        problems.append("Number of processes (-j) must be >= 1.")
    return problems


def plan_summary(state: FormState) -> PlanSummary:
    """Enumerate the segments + files a compile of *state* would generate.

    Resolves a ``--driver`` target to its model, then runs the same
    :func:`neml2.cli.aoti_export.plan_export_artifacts` planner the real compile
    uses -- no ``.pt2`` is produced, so this is cheap (one model load + structural
    analysis). Renames and the example batch shape are intentionally omitted: they
    change neither the segment structure nor the artifact filenames, and skipping
    them avoids spurious validation errors from a half-entered rename. The set is
    device-independent, so it is planned on ``cpu``. Raises if the model can't be
    loaded / planned; the caller (wizard) presents that as a muted note.
    """
    from .aoti_compile import driver_target_model  # noqa: PLC0415
    from .aoti_export import plan_export_artifacts  # noqa: PLC0415

    model_name = (
        driver_target_model(Path(state.input_file), state.name)
        if state.target == "driver"
        else state.name
    )
    plan = plan_export_artifacts(
        state.input_file,
        model_name,
        device="cpu",
        promoted=list(state.promoted),
        derivatives=tuple(state.derivatives),
        dynamic_batch=state.dynamic_batch,
    )
    segments = [(s.index, s.kind, s.basename, tuple(s.predicted_artifacts)) for s in plan.segments]
    files = (*plan.artifacts, f"{model_name}_aoti.i")
    return PlanSummary(segments=segments, files=files)


def list_section_entries(path: str | Path, section: str) -> list[tuple[str, str]]:
    """List ``(name, type)`` for every block under top-level ``[section]``.

    ``type`` is the block's ``type = ...`` field (e.g. ``"LinearIsotropicElasticity"``
    for a model, ``"TransientDriver"`` for a driver), or ``""`` if absent. Parses
    with ``nmhit`` only -- no model/driver is instantiated, so this works even
    when a block's ``type`` is a user extension that has not been imported.
    Mirrors the walk in :func:`neml2.cli.aoti_compile.driver_target_model`.
    """
    import nmhit  # noqa: PLC0415

    root = nmhit.parse_file(str(path), [], [])
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for top in root.children(nmhit.NodeType.Section):
        if top.path() != section:
            continue
        for child in top.children(nmhit.NodeType.Section):
            name = child.path()
            if name in seen:
                continue
            seen.add(name)
            entries.append((name, child.param_optional_str("type", "")))
    return entries


def list_model_names(path: str | Path) -> list[str]:
    """Names of the ``[Models]`` blocks in *path* (no instantiation)."""
    return [name for name, _ in list_section_entries(path, "Models")]


def list_driver_names(path: str | Path) -> list[str]:
    """Names of the ``[Drivers]`` blocks in *path* (no instantiation)."""
    return [name for name, _ in list_section_entries(path, "Drivers")]


__all__ = [
    "FormState",
    "IntrospectionForm",
    "PlanSummary",
    "build_compile_argv",
    "preview_command",
    "introspection_to_form",
    "validate_state",
    "plan_summary",
    "list_section_entries",
    "list_model_names",
    "list_driver_names",
]
