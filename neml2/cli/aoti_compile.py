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
                            [--device cpu|cuda] [--dtype float64|float32]
                            [-p|--parameter NAME ...]

By default every parameter and buffer in the source model is folded into the
exported graph as a constant. Each ``--parameter NAME`` flag promotes one
attribute to be a graph input -- mutable at runtime through
``aoti::Model::named_parameters()``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .aoti_export import export_model_for_aoti


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neml2-compile",
        description=(
            "Compile a NEML2 Python-native model to AOTI artifacts (.pt2 + "
            "metadata JSON) and emit a runnable HIT stub."
        ),
    )
    parser.add_argument("input", metavar="INPUT.i", type=Path, help="HIT input file path.")
    parser.add_argument(
        "--model",
        required=True,
        metavar="NAME",
        help="Model name in the input's [Models] section.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for the compiled artifacts (default: ./aoti/<NAME>/).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Target device for the artifact (baked at export time).",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float64", "float32"],
        help="Floating-point dtype for the artifact (baked at export time).",
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
    return parser


def emit_aoti_stub(
    original_hit: Path,
    model_name: str,
    meta_path: Path,
    stub_path: Path,
) -> None:
    """Read *original_hit*, replace the ``[Models]/<model_name>`` block with
    an AOTIModel shim pointing at *meta_path*, write the result to *stub_path*.

    The meta path is recorded relative to *stub_path*'s directory so the stub
    is movable as a unit. Other top-level sections (``[Drivers]``,
    ``[Settings]``, ``[Tensors]``, ``[Tensors]``, ...) are copied through
    verbatim from the original.
    """
    import nmhit  # noqa: PLC0415

    root = nmhit.parse_file(original_hit, [], [])

    # HIT lets a section name (e.g. ``[Models]``) appear in multiple
    # top-level blocks; nmhit doesn't auto-merge them. Walk every top-level
    # ``[Models]`` block to find the one that holds ``model_name``.
    # Existence check is by name only -- do NOT hold a Python reference to
    # the original sub-block past the remove_child call below. nmhit's
    # Python wrappers don't tolerate their underlying node being freed;
    # a kept reference triggers a hard nanobind crash on GC.
    models_blocks = [top for top in root.children(nmhit.NodeType.Section) if top.path() == "Models"]
    if not models_blocks:
        raise ValueError(f"{original_hit} has no [Models] section.")
    models = None
    all_names: list[str] = []
    for block in models_blocks:
        names = [c.path() for c in block.children(nmhit.NodeType.Section)]
        all_names.extend(names)
        if model_name in names:
            models = block
            break
    if models is None:
        raise ValueError(
            f"{original_hit} has no [Models/{model_name}] block. "
            f"--model must name an existing top-level entry. "
            f"Available: {sorted(set(all_names))}"
        )

    # Build the path the stub will refer to: relative to the stub's directory.
    stub_dir = stub_path.parent.resolve()
    meta_abs = meta_path.resolve()
    try:
        rel = os.path.relpath(meta_abs, start=stub_dir)
    except ValueError:
        # Different drives (Windows) or otherwise no common root -- fall back
        # to the absolute path. The stub stays valid, just not movable.
        rel = str(meta_abs)
    # HIT path strings live more naturally with an explicit "./" prefix when
    # they're relative -- otherwise readers can mistake them for type names.
    if not (rel.startswith("./") or rel.startswith("../") or os.path.isabs(rel)):
        rel = "./" + rel

    # Replace: remove the original block from [Models] by relpath (nmhit's
    # remove_child takes a path string), then add the shim with the same name.
    models.remove_child(model_name)
    shim = nmhit.Section(model_name)
    shim.add_child(nmhit.Field("type", "AOTIModel"))
    shim.add_child(nmhit.Field("meta", rel))
    models.add_child(shim)

    # Best-effort cleanup: if the original input set [Settings] aoti_mode =
    # true (no longer valid form), drop it. The stub is meant to
    # parse cleanly under the new C++ Factory which doesn't know aoti_mode.
    settings = root.find("Settings")
    if settings is not None:
        for key in ("aoti_mode", "aoti_cache_dir", "aoti_device"):
            if settings.find(key) is not None:
                settings.remove_child(key)

    header = (
        f"# Auto-generated by neml2-compile from {original_hit.name}.\n"
        f"# Drop-in replacement for the original [{model_name}] model.\n"
        f"# Do not edit; regenerate via `neml2-compile`.\n"
        f"\n"
    )
    stub_path.write_text(header + root.render())


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    # Trailing tokens are forwarded to the HIT parser as overrides
    # (e.g. ``Models/elasticity/E:=210000``), matching the convention the
    # other neml2-* CLIs use.
    args, additional_args = parser.parse_known_args(argv)

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir: Path = (args.output_dir or Path.cwd() / "aoti" / args.model).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        meta = export_model_for_aoti(
            input_path,
            args.model,
            output_dir,
            device=args.device,
            dtype=args.dtype,
            promoted=set(args.parameter),
            additional_args=tuple(additional_args),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error compiling '{args.model}': {exc}", file=sys.stderr)
        return 1

    meta_path = output_dir / f"{args.model}_meta.json"
    stub_path = output_dir / f"{args.model}_aoti.i"
    try:
        emit_aoti_stub(input_path, args.model, meta_path, stub_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error emitting stub: {exc}", file=sys.stderr)
        return 1

    n_promoted = len(args.parameter)
    bake_note = "fully baked" if n_promoted == 0 else f"{n_promoted} promoted parameter(s)"

    def _rel(p: Path) -> Path:
        return p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p

    print(f"Compiled '{args.model}' ({meta['type']}, {bake_note}) -> {output_dir}")
    print(f"  metadata: {_rel(meta_path)}")
    print(f"  stub:     {_rel(stub_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "emit_aoti_stub"]
