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

"""``neml2-compile`` -- the canonical (and only) path that produces an AOTI
artifact from a NEML2 input file.

Wraps :func:`neml2.cli.aoti_export.export_model_for_aoti` and emits a
runnable ``.i`` stub alongside the ``.pt2`` segments and ``_meta.json``. The
stub is the original input with the named ``[Models]/<name>`` block surgically
replaced by an AOTIModel shim pointing at the metadata file.

Usage:

    neml2-compile <input.i> --model <name>
                            [--output-dir <dir>]
                            [--device cpu|cuda [cpu|cuda ...]] [--dtype float64|float32]
                            [-p|--parameter NAME ...]

When more than one ``--device`` is given, a complete artifact is emitted per
device into a device-named subfolder (``<output-dir>/cpu/``, ``.../cuda/``).

By default every parameter and buffer in the source model is folded into the
exported graph as a constant. Each ``--parameter NAME`` flag promotes one
attribute to be a graph input -- mutable at runtime through
``aoti::Model::named_parameters()``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._extensions import add_load_argument, load_user_extensions
from .aoti_export import export_model_multidevice


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neml2-compile",
        description=(
            "Compile a NEML2 Python-native model to AOTI artifacts (.pt2 + "
            "metadata JSON) and emit a runnable HIT stub."
        ),
    )
    # `input` is always required (with --interactive too: the user picks the
    # file via shell completion, and the wizard takes it from there). Exactly one
    # of --model/--driver is required only for a normal compile -- the wizard
    # asks -- so the group is left optional here and enforced in main().
    parser.add_argument("input", metavar="INPUT.i", type=Path, help="HIT input file path.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "Launch an interactive wizard that introspects INPUT.i and walks you "
            "through the options (model/driver, promotable parameters, derivative "
            "pairs, devices, ...), then runs the resulting compile. Requires the "
            "'questionary' package."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=False)
    target.add_argument(
        "--model",
        metavar="NAME",
        help=(
            "Compile the named [Models] block. The emitted stub contains only "
            "the AOTIModel shim for this model -- no [Drivers] block. Pair with "
            "a custom driver in a separate file when you want to run it."
        ),
    )
    target.add_argument(
        "--driver",
        metavar="NAME",
        help=(
            "Compile whatever model the named [Drivers] block targets (via its "
            "`model = '...'` field). The stub keeps the named driver so the "
            "result is runnable as-is via `neml2-run`. Use this when you want "
            "a self-contained, drop-in replacement for the original driver."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Collection dir for the compiled artifacts (default: ./aoti/). Each "
            "model lands in <DIR>/<NAME>/<device>/ with a standalone "
            "<DIR>/<NAME>_aoti.i stub next to it."
        ),
    )
    parser.add_argument(
        "--device",
        nargs="+",
        default=["cpu"],
        choices=["cpu", "cuda"],
        metavar="DEVICE",
        help=(
            "Target device(s) for the artifact, baked at export time. Accepts "
            "more than one (e.g. --device cpu cuda): one complete artifact is "
            "emitted per device into a subfolder named by the device "
            "(<output-dir>/cpu/, <output-dir>/cuda/), ready for a multi-device "
            "dispatcher to load."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float64", "float32"],
        help="Floating-point dtype for the artifact (baked at export time).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Compile independent segments in parallel using up to N worker "
            "processes (spawn). Default 1 (serial). Only a multi-segment model "
            "(a ComposedModel containing an ImplicitUpdate) benefits; "
            "single-segment models ignore N, and N is capped at the number of "
            "segments (one task per segment). With --device cuda each worker "
            "spawns its own CUDA context and invokes nvcc -- watch memory."
        ),
    )
    parser.add_argument(
        "-p",
        "--parameter",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Promote a parameter or buffer (fully-qualified dotted name) to "
            "runtime-flexible status -- it becomes a graph input rather than "
            "a baked constant. Repeatable. With no -p flags the model is "
            "fully baked."
        ),
    )
    parser.add_argument(
        "-d",
        "--derivative",
        action="append",
        default=[],
        metavar="OUT:IN",
        help=(
            "Compile the derivative (Jacobian / JVP) graph for the OUT:IN "
            "output-input pair. Repeatable. Omit a side to select all on that "
            "side: 'stress:strain' is one pair, 'stress:' all inputs of stress, "
            "':strain' all outputs wrt strain, ':' all pairs. With no -d flags "
            "NO derivative graphs are compiled (smallest artifact) and the "
            "runtime jvp()/jacobian() raise until recompiled with -d."
        ),
    )
    parser.add_argument(
        "--example-batch-shape",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Example shape used when tracing. Two forms:\n"
            "  --example-batch-shape '(2,)'          (uniform across all inputs)\n"
            "  --example-batch-shape strain='(2;100)' --example-batch-shape temperature='(2,)'\n"
            "                                        (per-variable; repeatable)\n"
            "The semicolon delimits dynamic-batch from sub-batch axes -- '(2;100)' "
            "means dyn=(2,), sub=(100,). Overrides [Settings]/example_batch_shape."
        ),
    )
    parser.add_argument(
        "--dynamic-batch",
        dest="dynamic_batch",
        action="store_true",
        default=None,
        help=(
            "Compile a dynamic-batch artifact: the leading batch dim of the "
            "example shape becomes a runtime-variable torch.export Dim. "
            "Default unless overridden by [Settings]/dynamic_batch in the input."
        ),
    )
    parser.add_argument(
        "--no-dynamic-batch",
        dest="dynamic_batch",
        action="store_false",
        help=(
            "Pin the batch dim at the example shape. Required when a baked "
            "parameter has rank>=1 and would specialize the dynamic dim."
        ),
    )
    parser.add_argument(
        "--rename-input",
        action="append",
        default=[],
        metavar="ORIG:NEW",
        help=(
            "Rename an input variable at the compiled artifact's boundary: a "
            "downstream consumer sees NEW while the model's internal wiring keeps "
            "ORIG. Repeatable. Only valid with --model (not --driver)."
        ),
    )
    parser.add_argument(
        "--rename-output",
        action="append",
        default=[],
        metavar="ORIG:NEW",
        help=(
            "Rename an output variable at the boundary (the consumer sees NEW; "
            "the internals keep ORIG). Repeatable. Only valid with --model."
        ),
    )
    parser.add_argument(
        "--rename-parameter",
        action="append",
        default=[],
        metavar="ORIG:NEW",
        help=(
            "Rename a promoted parameter at the boundary (the consumer sees NEW; "
            "the internals keep the qualified ORIG). ORIG must be a promoted "
            "parameter (see -p). Repeatable. Only valid with --model."
        ),
    )
    add_load_argument(parser)
    return parser


def compile_and_emit_stub(
    input_path: Path,
    output_dir: Path,
    *,
    driver: str | None = None,
    model: str | None = None,
    device: str = "cpu",
    dtype: str = "float64",
    promoted: set[str] | None = None,
    example_batch_shape=None,
    dynamic_batch: bool | None = None,
    derivatives: tuple[str, ...] = (),
    renames: dict[str, dict[str, str]] | None = None,
    pre: tuple[str, ...] = (),
    additional_args: tuple[str, ...] = (),
    jobs: int = 1,
    progress_cb=None,
) -> Path:
    """Compile + emit stub in one call; return the stub path.

    Picks the model to compile from either ``driver`` (look up the driver
    block's ``model = '...'`` field) or ``model`` (compile that model
    directly). Exactly one must be supplied. Mirrors the
    ``neml2-compile`` CLI surface for callers who don't want to shell out
    to a subprocess.

    The intermediate ``model_name`` and the layout (``<output_dir>/<model>/<device>/``
    artifacts + a standalone ``<output_dir>/<model>_aoti.i`` stub) live entirely
    inside this function -- callers just run the returned stub path.

    *jobs* (> 1) compiles independent segments concurrently; *progress_cb*, if
    given, is invoked with the bare filename of every generated file (each
    ``.pt2``, the ``_meta.json``, and finally the ``_aoti.i`` stub).
    """
    # Local import keeps the module import-light when only the helpers
    # below are needed.
    from neml2.cli.aoti_export import export_model_for_aoti  # noqa: PLC0415

    if (driver is None) == (model is None):
        raise ValueError("compile_and_emit_stub: pass exactly one of `driver` or `model`")
    if driver is not None and renames and any(renames.values()):
        raise ValueError(
            "compile_and_emit_stub: boundary renames are only supported with "
            "`model` (not `driver`): the bundled [Drivers] block is wired to the "
            "original names."
        )
    model_name = driver_target_model(input_path, driver) if driver else model
    assert model_name is not None  # for type narrowing

    artifact_dir = output_dir / model_name
    device_dir = artifact_dir / device
    device_dir.mkdir(parents=True, exist_ok=True)
    export_model_for_aoti(
        input_path,
        model_name,
        device_dir,
        device=device,
        dtype=dtype,
        promoted=promoted or set(),
        example_batch_shape=example_batch_shape,
        dynamic_batch=dynamic_batch,
        derivatives=derivatives,
        renames=renames,
        pre=pre,
        additional_args=additional_args,
        jobs=jobs,
        progress_cb=progress_cb,
    )
    stub_path = output_dir / f"{model_name}_aoti.i"
    emit_aoti_stub(
        input_path,
        model_name,
        artifact_dir,
        stub_path,
        keep_drivers=(driver is not None),
    )
    if progress_cb is not None:
        progress_cb(stub_path.name)
    return stub_path


def _parse_rename_pairs(specs: list[str], flag: str) -> dict[str, str]:
    """Parse ``ORIG:NEW`` rename tokens into an ``{orig: new}`` map.

    Splits on the FIRST colon: variable names carry no colon and a qualified
    promoted-parameter name (e.g. ``elasticity.E``) uses dots, so the first
    colon is unambiguously the ORIG/NEW separator. Raises ``ValueError`` on a
    malformed token (missing colon, empty side) or a duplicate ORIG.
    """
    out: dict[str, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"{flag}: expected ORIG:NEW, got {spec!r} (missing ':').")
        orig, new = (part.strip() for part in spec.split(":", 1))
        if not orig or not new:
            raise ValueError(f"{flag}: expected ORIG:NEW with both sides non-empty, got {spec!r}.")
        if orig in out:
            raise ValueError(f"{flag}: duplicate rename for {orig!r}.")
        out[orig] = new
    return out


def _renames_from_args(args) -> dict[str, dict[str, str]]:
    """Assemble the ``{inputs, outputs, parameters}`` rename map from parsed CLI args."""
    return {
        "inputs": _parse_rename_pairs(args.rename_input, "--rename-input"),
        "outputs": _parse_rename_pairs(args.rename_output, "--rename-output"),
        "parameters": _parse_rename_pairs(args.rename_parameter, "--rename-parameter"),
    }


def driver_target_model(original_hit: Path, driver_name: str) -> str:
    """Return the model name the ``[Drivers]/<driver_name>/model`` field points at.

    Used by ``--driver Y`` mode to derive ``--model X`` automatically: the
    driver block already declares the model it runs, so there's no need for
    the caller (or a parallel registry like ``benchmark/sweep.py``'s
    ``_MODEL_OVERRIDES``) to repeat the mapping.
    """
    import nmhit  # noqa: PLC0415

    root = nmhit.parse_file(original_hit, [], [])
    drivers_blocks = [
        top for top in root.children(nmhit.NodeType.Section) if top.path() == "Drivers"
    ]
    available: list[str] = []
    for block in drivers_blocks:
        for child in block.children(nmhit.NodeType.Section):
            available.append(child.path())
            if child.path() == driver_name:
                model_field = child.find("model")
                if model_field is None:
                    raise ValueError(
                        f"{original_hit}: [Drivers/{driver_name}] has no `model` field; "
                        f"cannot infer what to compile."
                    )
                return child.param_str("model")
    raise ValueError(
        f"{original_hit}: no [Drivers/{driver_name}] block. "
        f"Available drivers: {sorted(set(available))}."
    )


def emit_aoti_stub(
    original_hit: Path,
    model_name: str,
    artifact_dir: Path,
    stub_path: Path,
    *,
    keep_drivers: bool = False,
) -> None:
    """Write a minimal AOTI stub at *stub_path*.

    The stub is a standalone file that sits next to (not inside) the
    *artifact_dir* -- the per-device artifact folder holding one ``<device>/``
    subfolder per compiled device. The shim points at it via an absolute
    ``artifact_path``; the loader (Python shim or C++ ``load_model``) resolves
    ``<artifact_path>/<device>/<model>_meta.json`` for the device it runs on.

    The stub contains exactly what's needed to load the compiled artifact:

    * a single ``[Models]`` block holding only the ``AOTIModel`` shim for
      ``<model_name>``. All sibling entries (and any other top-level
      ``[Models]`` blocks the original file may have had) are dropped --
      they're unreachable from the shim at runtime, and keeping a
      ``ComposedModel`` whose child is now an ``AOTIModel`` shim would
      break ``ComposedModel``'s resolver at load time (the shim has no
      ``provided_items``);
    * ``[Settings]`` from the original (small + safe; ``aoti_*`` keys are
      stripped since they're no longer valid for the v3 factory);
    * ``[Tensors]`` from the original verbatim -- a driver, if kept, may
      reference any entry and entries reference each other transitively;
      pruning by reachability buys little and risks dropping something
      needed;
    * ``[Drivers]`` from the original verbatim when ``keep_drivers=True``
      (``--driver`` mode); drivers can reference other drivers (e.g.
      ``TransientRegression`` wrapping a ``TransientDriver``), so we
      mirror the ``[Tensors]`` policy and keep them all. Dropped entirely
      when ``keep_drivers=False`` (``--model`` mode).

    ``[Data]`` and ``[EquationSystems]`` are dropped in both modes -- their
    state was baked into the ``.pt2`` at compile time (the implicit Newton
    system, any ``[Data]``-typed constants). For an implicit model a MINIMAL
    ``[Solvers]`` block is carried (schema v4+): just the implicit model's
    solver, and only its honored convergence / line-search knobs. The shim
    references it via a ``solver = <name>`` field and forwards those settings to
    the C++ runtime at load. The ``linear_solver`` field is dropped on purpose
    -- it is baked into the compiled step/IFT graphs, so editing it in the stub
    would have no effect; omitting it keeps the stub free of inert knobs.

    The artifact folder is recorded as an absolute ``artifact_path`` -- the
    stub is standalone, so the artifacts are not relocatable without a recompile.

    Implementation note: we build a fresh ``nmhit.Root`` and clone the
    sections we want into it, rather than mutating the parsed input. nmhit
    surfaces a Python wrapper around each node; calling ``remove_child``
    on the original root invalidates any other wrapper that pointed at the
    removed subtree, and any subsequent access crashes nanobind on GC.
    Cloning sidesteps the lifetime entanglement entirely.
    """
    import nmhit  # noqa: PLC0415

    DROPPED = {"Data", "EquationSystems"}

    src_root = nmhit.parse_file(original_hit, [], [])

    # Discover which (single) [Models] block holds our target, and which solver
    # the implicit Newton solve should be configured with (schema v4+: solver
    # config is carried in the stub, not baked). We mutate nothing in src_root
    # -- discovery is read-only -- and we don't hold any node references past
    # this scan.
    target_models_idx: int | None = None
    available_in_block: list[list[str]] = []
    target_solver = ""
    implicit_solvers: list[str] = []
    models_seen = 0
    for top in src_root.children(nmhit.NodeType.Section):
        if top.path() != "Models":
            continue
        names: list[str] = []
        for c in top.children(nmhit.NodeType.Section):
            names.append(c.path())
            csolver = c.param_optional_str("solver", "")
            if c.param_optional_str("type", "") == "ImplicitUpdate" and csolver:
                if csolver not in implicit_solvers:
                    implicit_solvers.append(csolver)
            if c.path() == model_name and csolver:
                target_solver = csolver
        available_in_block.append(names)
        if model_name in names and target_models_idx is None:
            target_models_idx = models_seen
        models_seen += 1
    # Prefer the compiled model's own solver (direct ImplicitUpdate); fall back
    # to the first implicit solver in the graph (composed model). Forward-only
    # models leave this empty -> the shim applies the C++ defaults.
    solver_name = target_solver or (implicit_solvers[0] if implicit_solvers else "")
    if len({*implicit_solvers}) > 1 and not target_solver:
        print(
            f"neml2-compile: multiple implicit solvers {implicit_solvers} found; "
            f"the AOTI stub configures the runtime with {solver_name!r} for all "
            "implicit segments.",
            file=sys.stderr,
        )
    if target_models_idx is None:
        all_names = [n for block in available_in_block for n in block]
        raise ValueError(
            f"{original_hit} has no [Models/{model_name}] block. "
            f"Available: {sorted(set(all_names))}"
        )

    # Absolute pointer at the per-device artifact folder. Absolute (not
    # relative) by design: the stub is standalone and lives outside the folder,
    # so moving the artifacts requires recompiling or editing this path.
    #
    # Single-quote the value so the path round-trips through the HIT lexer
    # verbatim -- an unquoted scalar rejects characters that are legal in a
    # filesystem path (Windows backslashes, spaces), whereas a single-quoted
    # value is raw. This matches the format the reader documents and expects
    # (see the ``artifact_path = '...'`` examples in ``neml2/aoti/_shim.py`` and
    # ``doc/content/references/aoti_packages.md``).
    artifact_abs = str(Path(artifact_dir).resolve())

    # Build the cleaned [Models] block once: one shim, no siblings.
    cleaned_models = nmhit.Section("Models")
    shim = nmhit.Section(model_name)
    shim.add_child(nmhit.Field("type", "AOTIModel"))
    shim.add_child(nmhit.Field("artifact_path", f"'{artifact_abs}'"))
    if solver_name:
        # The carried (minimal) [Solvers] block configures the implicit Newton
        # solve at load -- convergence / line-search knobs only.
        shim.add_child(nmhit.Field("solver", solver_name))
    cleaned_models.add_child(shim)

    # Walk the source top-level children IN ORDER and clone each into the
    # new root. Skip dropped sections; for [Models] only emit the cleaned
    # block once (at the index of the original target so surrounding
    # comments / blanks land near it).
    new_root = nmhit.Root()
    models_emitted = False
    models_seen = 0
    for top in src_root.children():
        t = top.type()
        # Preserve top-level comments + blank lines so the stub still
        # reads like the original where it overlaps.
        if t != nmhit.NodeType.Section:
            new_root.add_child(top.clone())
            continue

        name = top.path()
        if name in DROPPED:
            continue
        if name == "Drivers" and not keep_drivers:
            continue
        if name == "Solvers":
            # Carry a MINIMAL block: only the implicit model's solver, and only
            # its honored knobs. The linear solver is baked into the compiled
            # step/IFT graphs, so editing it here would be a silent no-op --
            # drop the `linear_solver` field (and the sub-solver blocks it would
            # reference) so the stub exposes only settings that take effect.
            if not solver_name:
                continue  # forward-only model: no solver to carry
            min_solvers = nmhit.Section("Solvers")
            for c in top.children(nmhit.NodeType.Section):
                if c.path() != solver_name:
                    continue
                cloned = c.clone()
                if cloned.find("linear_solver") is not None:
                    cloned.remove_child("linear_solver")
                min_solvers.add_child(cloned)
            new_root.add_child(min_solvers)
            continue
        if name == "Models":
            if models_seen == target_models_idx and not models_emitted:
                new_root.add_child(cleaned_models)
                models_emitted = True
            # Any other [Models] block: drop entirely.
            models_seen += 1
            continue
        if name == "Settings":
            cloned = top.clone()
            for key in ("aoti_mode", "aoti_cache_dir", "aoti_device"):
                if cloned.find(key) is not None:
                    cloned.remove_child(key)
            new_root.add_child(cloned)
            continue
        # Default: [Tensors], [Drivers] when kept, anything else we don't
        # explicitly know -- copy through.
        new_root.add_child(top.clone())

    mode_note = "driver-mode" if keep_drivers else "model-mode"
    header = (
        f"# Auto-generated by neml2-compile from {original_hit.name} ({mode_note}).\n"
        f"# Drop-in replacement for the original [{model_name}] model.\n"
        f"# Do not edit; regenerate via `neml2-compile`.\n"
        f"\n"
    )
    # encoding="utf-8": model/variable names may contain non-ASCII characters
    # (e.g. Greek symbols like theta). Without an explicit encoding, Windows
    # writes with the locale codepage (cp1252) and raises UnicodeEncodeError.
    stub_path.write_text(header + new_root.render(), encoding="utf-8")


def _parse_example_batch_shape_cli(entries: list[str]) -> dict[str, str] | str | None:
    """Reduce ``--example-batch-shape`` CLI entries to the kwarg form
    :func:`~neml2.cli.aoti_export.export_model_for_aoti` accepts.

    A single uniform entry (``'(2,)'``) returns the string verbatim. One or
    more per-variable entries (``'strain=(2;100)'``) return a dict. Mixing
    forms is an error.
    """
    if not entries:
        return None
    uniform: list[str] = []
    per_var: dict[str, str] = {}
    for raw in entries:
        if "=" in raw and not raw.lstrip().startswith("("):
            name, _, spec = raw.partition("=")
            name = name.strip()
            spec = spec.strip()
            if not name or not spec:
                raise ValueError(
                    f"--example-batch-shape {raw!r}: expected name=spec (e.g. strain='(2;100)')."
                )
            per_var[name] = spec
        else:
            uniform.append(raw)
    if uniform and per_var:
        raise ValueError("--example-batch-shape: cannot mix uniform and per-variable forms.")
    if len(uniform) > 1:
        raise ValueError(
            "--example-batch-shape: uniform form accepts a single spec, got "
            f"{len(uniform)}: {uniform}."
        )
    return uniform[0] if uniform else per_var


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    # Trailing tokens are forwarded to the HIT parser as overrides
    # (e.g. ``Models/elasticity/E:=210000``), matching the convention the
    # other neml2-* CLIs use.
    args, additional_args = parser.parse_known_args(argv)

    # Interactive mode: hand off to the wizard, which collects the remaining
    # options (model/driver, parameters, derivatives, ...) for the given INPUT.i
    # and then re-runs this command with the assembled argv. `questionary` is
    # imported lazily so the normal compile path never depends on it.
    if args.interactive:
        try:
            import questionary  # noqa: F401, PLC0415
        except ImportError:
            print(
                "neml2-compile --interactive needs the 'questionary' package.\n"
                "Install it with:  pip install questionary",
                file=sys.stderr,
            )
            return 1
        from ._compile_wizard import run_wizard  # noqa: PLC0415

        return run_wizard(input_file=str(args.input), initial_load=tuple(args.load))

    # Non-interactive: exactly one of --model/--driver is required (argparse
    # leaves the group optional so --interactive can omit it).
    if (args.model is None) == (args.driver is None):
        parser.error("exactly one of --model / --driver is required (or use --interactive)")

    # Boundary renames (shallow). Parse early, and reject in --driver mode: the
    # driver stub bakes the original [Drivers] block (wired to the original
    # names) into the runnable stub, which the renamed interface would break.
    try:
        renames = _renames_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.driver is not None and any(renames.values()):
        parser.error(
            "--rename-input / --rename-output / --rename-parameter are only "
            "supported with --model (not --driver): the bundled [Drivers] block "
            "is wired to the original names. Compile with --model and pair the "
            "renamed artifact with your own consumer."
        )

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        load_user_extensions(args.load)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Resolve the (model_name, keep_driver) pair from --model vs --driver.
    # argparse mutual exclusion already guarantees at most one is set.
    if args.driver is not None:
        try:
            model_name = driver_target_model(input_path, args.driver)
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        keep_driver: str | None = args.driver
    else:
        model_name = args.model
        keep_driver = None

    # `--output-dir` is the collection dir (default ./aoti). Each model lands in
    # <output-dir>/<model>/<device>/ with one standalone stub next to it at
    # <output-dir>/<model>_aoti.i.
    output_dir: Path = (args.output_dir or Path.cwd() / "aoti").resolve()
    artifact_dir = output_dir / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        example_batch_shape = _parse_example_batch_shape_cli(args.example_batch_shape)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    # One complete artifact per device, each in its own device-named subfolder
    # (<model>/cpu/, <model>/cuda/). De-dupe while preserving order so
    # `--device cpu cpu` doesn't compile twice.
    devices = list(dict.fromkeys(args.device))

    # Enumerate every file the compile will generate so progress can report
    # [k/N]. The artifact set is device-independent, so a single plan (on cpu, to
    # avoid initializing CUDA in the parent) sizes the whole multi-device run;
    # the trailing +1 is the standalone `_aoti.i` stub emitted once at the end.
    from neml2.cli.aoti_export import plan_export_artifacts  # noqa: PLC0415

    try:
        plan = plan_export_artifacts(
            input_path,
            model_name,
            device="cpu",
            promoted=set(args.parameter),
            example_batch_shape=example_batch_shape,
            dynamic_batch=args.dynamic_batch,
            derivatives=tuple(args.derivative),
            renames=renames,
            additional_args=tuple(additional_args),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error planning '{model_name}': {exc}", file=sys.stderr)
        return 1
    total_files = plan.total * len(devices) + 1  # per-device artifacts + the stub

    _progress = {"k": 0}

    def progress_cb(name: str) -> None:
        _progress["k"] += 1
        print(f"[{_progress['k']}/{total_files}] {name}", file=sys.stderr)

    if args.jobs > 1 and "cuda" in devices:
        print(
            f"neml2-compile: warning: -j{args.jobs} with --device cuda spawns "
            f"{args.jobs} worker processes, each initializing its own CUDA context "
            "and invoking nvcc; watch GPU/host memory (consider -j1 for cuda).",
            file=sys.stderr,
        )

    # Compile every device, parallelizing across the full (device x segment) grid
    # (jobs bounds the workers across ALL cells, so multiple devices compile
    # concurrently). progress_cb receives device-tagged names from the orchestrator.
    try:
        metas = export_model_multidevice(
            input_path,
            model_name,
            artifact_dir,
            devices,
            dtype=args.dtype,
            promoted=set(args.parameter),
            example_batch_shape=example_batch_shape,
            dynamic_batch=args.dynamic_batch,
            derivatives=tuple(args.derivative),
            renames=renames,
            additional_args=tuple(additional_args),
            jobs=args.jobs,
            progress_cb=progress_cb,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error compiling '{model_name}': {exc}", file=sys.stderr)
        return 1
    compiled = [(device, artifact_dir / device / f"{model_name}_meta.json") for device in devices]
    meta = next(iter(metas.values()))  # envelope is identical across devices

    # One standalone stub next to the artifact folder, pointing at it.
    stub_path = output_dir / f"{model_name}_aoti.i"
    try:
        emit_aoti_stub(
            input_path,
            model_name,
            artifact_dir,
            stub_path,
            keep_drivers=(keep_driver is not None),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error emitting stub: {exc}", file=sys.stderr)
        return 1
    progress_cb(stub_path.name)

    n_promoted = len(args.parameter)
    bake_note = "fully baked" if n_promoted == 0 else f"{n_promoted} promoted parameter(s)"
    driver_note = f", driven by [{keep_driver}]" if keep_driver else ""
    n_deriv = len(meta.get("derivatives", []))
    n_param_deriv = len(meta.get("parameter_derivatives", []))
    if n_deriv == 0 and n_param_deriv == 0:
        deriv_note = ", no derivative graphs (use -d to compile jvp/jacobian)"
    else:
        parts = []
        if n_deriv:
            parts.append(f"{n_deriv} derivative pair(s)")
        if n_param_deriv:
            parts.append(f"{n_param_deriv} parameter-derivative pair(s)")
        deriv_note = ", " + ", ".join(parts)

    def _rel(p: Path) -> Path:
        return p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p

    dev_list = ", ".join(device for device, _ in compiled)
    print(
        f"Compiled '{model_name}' ({meta['type']}, {bake_note}{driver_note}{deriv_note}) "
        f"for [{dev_list}] -> {_rel(artifact_dir)}"
    )
    print(f"  stub: {_rel(stub_path)}")
    for device, meta_path in compiled:
        print(f"  {device}: {_rel(meta_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "compile_and_emit_stub", "driver_target_model", "emit_aoti_stub"]
