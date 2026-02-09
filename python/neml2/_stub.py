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

from __future__ import annotations

import importlib.util
import pkgutil
import subprocess
import sys
from pathlib import Path

import neml2


def _discover_modules() -> list[str]:
    modules = [neml2.__name__]
    prefix = f"{neml2.__name__}."
    modules.extend(sorted(module.name for module in pkgutil.walk_packages(neml2.__path__, prefix)))
    return modules


def _generate_stub(*args: str) -> int:
    if importlib.util.find_spec("pybind11_stubgen") is None:
        print("Unable to generate stubs: `pybind11-stubgen` is not installed")
        return 1

    site_packages = Path(neml2.__path__[0]).resolve().parent
    cmd_base = [
        sys.executable,
        "-m",
        "pybind11_stubgen",
        "-o",
        str(site_packages),
    ]
    cmd_base.extend(args)

    for module in _discover_modules():
        print(f"Generating stubs for {module}")
        result = subprocess.run(cmd_base + [module])
        if result.returncode != 0:
            return result.returncode

    return 0
