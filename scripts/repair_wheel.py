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
          deps and must not be bundled).
  macOS:  pass-through (delocate is skipped; torch dylibs are not bundleable and
          plain macosx_* tags are accepted by PyPI).

Usage: python repair_wheel.py {wheel} {dest_dir}
"""

import shutil
import subprocess
import sys
from pathlib import Path


def _torch_excludes() -> list[str]:
    """Return --exclude flags for every .so in PyTorch's lib directory."""
    try:
        import torch

        lib_dir = Path(torch.__file__).parent / "lib"
        return [
            flag for f in lib_dir.iterdir() if ".so" in f.name for flag in ("--exclude", f.name)
        ]
    except ImportError:
        return []


def main() -> None:
    wheel = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])
    dest_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform.startswith("linux"):
        subprocess.run(
            ["auditwheel", "repair", "-w", str(dest_dir)] + _torch_excludes() + [str(wheel)],
            check=True,
        )
    else:
        # macOS: pass-through; delocate would try to bundle torch dylibs.
        shutil.copy(wheel, dest_dir / wheel.name)


if __name__ == "__main__":
    main()
