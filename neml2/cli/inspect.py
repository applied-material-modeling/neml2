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

"""``neml2-inspect`` -- native equivalent of ``src/tools/neml2-inspect.cxx``.

Summarizes a model's structure: its inputs / outputs (named typed variables),
its registered parameters and buffers (dtype + device + shape). Default
output is human-readable; ``--json`` emits a structured one-line JSON object
(``{"retcode": 0, ...}`` on success, ``{"retcode": 1, "error": "..."}`` on
failure). The exit code mirrors ``retcode``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any

from ..factory import load_input
from ._extensions import add_load_argument, load_user_extensions

if TYPE_CHECKING:
    from ..model import Model


def _model_to_dict(model: Model) -> dict[str, Any]:
    return {
        "name": type(model).__name__,
        "inputs": [{"name": name, "type": typ.__name__} for name, typ in model.input_spec.items()],
        "outputs": [
            {"name": name, "type": typ.__name__} for name, typ in model.output_spec.items()
        ],
        "parameters": [
            {
                "name": name,
                "dtype": str(p.dtype),
                "device": str(p.device),
                "shape": list(p.shape),
            }
            for name, p in model.named_parameters(recurse=True)
        ],
        "buffers": [
            {
                "name": name,
                "dtype": str(b.dtype),
                "device": str(b.device),
                "shape": list(b.shape),
            }
            for name, b in model.named_buffers(recurse=True)
        ],
    }


def _emit_human(data: dict[str, Any]) -> None:
    print(f"Model: {data['name']}")
    for section in ("inputs", "outputs", "parameters", "buffers"):
        items = data[section]
        print(f"\n{section.title()} ({len(items)}):")
        for item in items:
            extras = ", ".join(f"{k}={v}" for k, v in item.items() if k != "name")
            print(f"  {item['name']}: {extras}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neml2-inspect",
        description=(
            "Summarize the structure of a model. Trailing positional tokens are "
            "treated as HIT overrides (e.g. 'Models/elasticity/E:=210000')."
        ),
    )
    parser.add_argument("input", help="path to the input file")
    parser.add_argument("model", help="name of the model in the input file to inspect")
    parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        help="emit a structured JSON description on stdout instead of the human-readable format",
    )
    add_load_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    # ``parse_known_args`` separates declared options from any trailing HIT
    # overrides without REMAINDER greedily capturing ``--json``.
    args, additional_args = _build_parser().parse_known_args(argv)
    try:
        load_user_extensions(args.load)
        factory = load_input(args.input, additional_args=additional_args)
        model = factory.get_model(args.model)
    except Exception as exc:  # noqa: BLE001
        if args.json_mode:
            print(json.dumps({"retcode": 1, "error": str(exc)}))
            return 1
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    data = _model_to_dict(model)
    if args.json_mode:
        data["retcode"] = 0
        print(json.dumps(data))
    else:
        _emit_human(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
