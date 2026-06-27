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

"""``neml2-run`` -- native equivalent of ``src/tools/neml2-run.cxx``."""

from __future__ import annotations

import argparse
import sys

from ..factory import load_input
from ._extensions import add_load_argument, load_user_extensions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neml2-run",
        description=(
            "Run a driver from an input file. Trailing positional tokens are "
            "treated as HIT overrides (e.g. 'Models/elasticity/E:=210000')."
        ),
    )
    parser.add_argument("input", help="path to the input file")
    parser.add_argument("driver", help="name of the driver in the input file")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help=(
            "Set torch's default device before loading. Tensors built by "
            "[Tensors] Python expressions inherit this. Default: cpu."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float64", "float32"],
        help=(
            "Set torch's default dtype before loading. NEML2 models are "
            "uniformly float64 by convention; AOTI artifacts compile-pin "
            "their dtype and the runtime rejects silent coercion, so "
            "float64 is the safe default. Override only if you know the "
            "artifact was compiled with float32."
        ),
    )
    add_load_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    # ``parse_known_args`` separates the two declared positionals from any
    # trailing HIT-override tokens. Using REMAINDER would greedily capture
    # subsequent flags too.
    args, additional_args = _build_parser().parse_known_args(argv)

    # Set process-wide torch defaults BEFORE load_input. [Tensors] Python
    # expressions (``torch.tensor([...])``, ``torch.linspace(...)``, ...)
    # build their initial conditions on the active dtype/device. Without
    # this, a fresh process's default float32 would mismatch an AOTI
    # artifact compiled as float64 -- and the runtime correctly refuses
    # to silently coerce, surfacing a confusing dtype error.
    import torch  # noqa: PLC0415

    torch.set_default_dtype(getattr(torch, args.dtype))
    torch.set_default_device(args.device)

    try:
        load_user_extensions(args.load)
        factory = load_input(args.input, additional_args=additional_args)
        # ``run()`` writes its output to disk iff the driver's ``save_as`` is
        # set (TransientDriver); nothing CLI-specific to do here.
        factory.get_driver(args.driver).run()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
