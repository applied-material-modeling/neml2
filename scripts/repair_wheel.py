#!/usr/bin/env python3
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

"""
Post-build wheel repair invoked by cibuildwheel's repair-wheel-command:

  Linux:  run auditwheel excluding PyTorch .so files (they are declared runtime
          deps and must not be bundled), then inject pybind11 stubs.
  macOS:  skip delocate (torch dylibs are not bundleable; plain macosx_* tags
          are accepted by PyPI), then inject pybind11 stubs.

Usage: python repair_wheel.py {wheel} {dest_dir}
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _torch_excludes() -> list[str]:
    """Return --exclude flags for every .so in PyTorch's lib directory."""
    try:
        import torch

        lib_dir = Path(torch.__file__).parent / "lib"
        return [
            flag
            for f in lib_dir.iterdir()
            if ".so" in f.name
            for flag in ("--exclude", f.name)
        ]
    except ImportError:
        return []


def _generate_stubs(wheel: Path) -> list[Path]:
    """Install the wheel, run neml2-stub, and return the generated .pyi paths."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", str(wheel)],
        check=True,
    )
    import neml2
    from neml2._stub import _generate_stub

    _generate_stub()
    return list(Path(neml2.__path__[0]).glob("*.pyi"))


def _inject_stubs(wheel: Path, stubs: list[Path], dest_dir: Path) -> None:
    """Unpack wheel, copy .pyi files into neml2/, repack into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["wheel", "unpack", str(wheel), "--dest", tmp], check=True)
        pkg_dir = next(Path(tmp).iterdir())
        for s in stubs:
            shutil.copy(s, pkg_dir / "neml2" / s.name)
        subprocess.run(
            ["wheel", "pack", str(pkg_dir), "--dest-dir", str(dest_dir)],
            check=True,
        )


def main() -> None:
    wheel = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])

    stubs = _generate_stubs(wheel)

    if sys.platform.startswith("linux"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["auditwheel", "repair", "-w", tmp] + _torch_excludes() + [str(wheel)],
                check=True,
            )
            repaired = next(Path(tmp).glob("*.whl"))
            _inject_stubs(repaired, stubs, dest_dir)
    else:
        _inject_stubs(wheel, stubs, dest_dir)


if __name__ == "__main__":
    main()
