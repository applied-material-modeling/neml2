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

from .._accelerator import KNOWN_FAMILIES, parse_device_spec
from ..factory import load_input
from ._extensions import add_load_argument, load_user_extensions


def _device_arg(s: str) -> str:
    """argparse type: validate and normalize a --device spec to its canonical str.

    Accepts bare family names (``cpu``, ``cuda``, ``xpu``, ``hip``, ``mps``) and
    indexed forms (``cuda:1``, ``xpu:0``). Returns the canonical torch spelling
    (``str(torch.device(...))``) so ``torch.set_default_device`` and downstream
    consumers work with a string.
    """
    return str(parse_device_spec(s))


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
        type=_device_arg,
        metavar="DEVICE",
        help=(
            "Set torch's default device before loading. Tensors built by "
            f"[Tensors] Python expressions inherit this. Any of {list(KNOWN_FAMILIES)}, "
            "optionally with a device index (e.g. cuda:1, xpu:0). Default: cpu."
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
    parser = _build_parser()
    args, additional_args = parser.parse_known_args(argv)

    # Refuse a (device, dtype) combination torch itself does not support
    # (e.g. mps + float64: Apple's MPS backend has no fp64 path). Fail fast
    # here rather than crashing inside torch further down. Only the family
    # matters -- an indexed spec like "mps:0" narrows to the same family.
    from neml2._accelerator import is_compatible, parse_device_spec  # noqa: PLC0415

    _fam = parse_device_spec(args.device).type
    if not is_compatible(_fam, args.dtype):
        parser.error(
            f"unsupported (device, dtype) combination: {args.device} + "
            f"{args.dtype}. This accelerator does not support this dtype."
        )

    # Set process-wide torch defaults BEFORE load_input so [Tensors] Python
    # expressions (``torch.tensor([...])``, ``torch.linspace(...)``, ...) build
    # their initial conditions on the active dtype/device -- otherwise a fresh
    # process's float32 default would mismatch a float64 artifact's inputs, and
    # the runtime correctly refuses to silently coerce. (The AOTI <device>/<dtype>/
    # leaf itself is selected by the explicit device/dtype passed to load_input
    # below, not by these ambient defaults.)
    import torch  # noqa: PLC0415

    torch.set_default_dtype(getattr(torch, args.dtype))
    torch.set_default_device(args.device)

    try:
        load_user_extensions(args.load)
        # Pass device/dtype explicitly so a compiled (AOTIModel) block loads the
        # matching <device>/<dtype>/ leaf. The load APIs no longer read torch's
        # ambient defaults for that choice; the set_default_* above remains only so
        # [Tensors] Python expressions build their tensors on the active dtype/device.
        factory = load_input(
            args.input, additional_args=additional_args, device=args.device, dtype=args.dtype
        )
        # ``run()`` writes its output to disk iff the driver's ``save_as`` is
        # set (TransientDriver); nothing CLI-specific to do here.
        factory.get_driver(args.driver).run()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
