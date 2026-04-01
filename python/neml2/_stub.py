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

import importlib.machinery
import importlib.util
import pkgutil
from pathlib import Path

import neml2


def _is_extension_module(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return False
    return any(spec.origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def _discover_modules() -> list[str]:
    prefix = f"{neml2.__name__}."
    return sorted(
        module.name
        for module in pkgutil.walk_packages(neml2.__path__, prefix)
        if _is_extension_module(module.name)
    )


def _fixup_parameter_store_stubs(content: str) -> str:
    """Fix Tensor -> TensorLike in setter methods shared by Model and NonlinearSystem."""
    content = content.replace(
        "def __setattr__(self, arg0: str, arg1: neml2.Tensor) -> None:",
        "def __setattr__(self, arg0: str, arg1: neml2.TensorLike) -> None:",
    )
    content = content.replace(
        "def set_parameter(self, arg0: str, arg1: neml2.Tensor) -> None:",
        "def set_parameter(self, arg0: str, arg1: neml2.TensorLike) -> None:",
    )
    content = content.replace(
        "def set_parameters(self, arg0: dict[str, neml2.Tensor]) -> None:",
        "def set_parameters(self, arg0: dict[str, neml2.TensorLike]) -> None:",
    )
    return content


def _fixup_stubs(site_packages: Path) -> None:
    """Fix type annotations in generated stubs that pybind11-stubgen gets wrong.

    pybind11-stubgen is unaware of implicit conversions registered with
    py::implicitly_convertible, so it annotates setter methods as accepting
    only neml2.Tensor rather than the full neml2.TensorLike union.
    """
    for filename in ("core.pyi", "es.pyi"):
        pyi = site_packages / "neml2" / filename
        if not pyi.exists():
            continue

        with open(pyi) as f:
            content = f.read()

        content = _fixup_parameter_store_stubs(content)

        with open(pyi, "w") as f:
            f.write(content)


def _generate_stub(*args: str) -> int:
    if importlib.util.find_spec("pybind11_stubgen") is None:
        print("Unable to generate stubs: `pybind11-stubgen` is not installed")
        return 1

    import pybind11_stubgen

    site_packages = Path(neml2.__path__[0]).resolve().parent
    argv_base = ["-o", str(site_packages), *args]

    retcode = 0
    for module in _discover_modules():
        print(f"Generating stubs for {module}")
        try:
            pybind11_stubgen.main(argv_base + [module])
        except SystemExit as e:
            if e.code:
                print(f"Failed to generate stubs for {module}")
                retcode = int(e.code)

    _fixup_stubs(site_packages)

    return retcode
