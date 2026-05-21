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

import importlib
import typing
from pathlib import Path

# Submodules and compiled bits. Star imports from the compiled submodules
# are intentional re-exports — see the `**/__init__.py` entry in
# `[tool.ruff.lint.per-file-ignores]`.
from . import crystallography, pyzag, reader
from .core import *
from .es import *
from .tensors import *

# `neml2.math` registers `.norm()`, `.dynamic_linspace`, and other math
# operators onto every tensor class at import time via
# `py::module_::import("neml2.tensors").attr(T)`, so `.tensors` MUST already
# be loaded when `.math` runs its module init. Imported via `importlib` (not
# `from .math import *`) so ruff's isort cannot reorder it ahead of
# `.tensors`, and so that `neml2.math.Number` (the only public symbol) does
# not shadow the `Number` alias defined below.
importlib.import_module(".math", __name__)

# Determine version
version_file = Path(__file__).parent / "version"
if version_file.exists():
    __version__ = version_file.read_text().strip()
else:
    __version__ = "unknown"

# Determine hash
hash_file = Path(__file__).parent / "hash"
if hash_file.exists():
    __hash__ = hash_file.read_text().strip()
else:
    __hash__ = "unknown"

Number = int | float | bool

# pybind11-stubgen generates incorrect type annotations for Unions
# so unfortunately we need to maintain this list (and keep `typing.Union` —
# the `X | Y` form trips the same stub-gen bug).
# see issue https://github.com/sizmailov/pybind11-stubgen/issues/276
TensorLike = typing.Union[  # noqa: UP007
    Vec,
    Rot,
    WR2,
    R2,
    Scalar,
    SR2,
    R3,
    SFR3,
    R4,
    SFFR4,
    WFFR4,
    SSR4,
    SWR4,
    WSR4,
    WWR4,
    Quaternion,
    MillerIndex,
    Tensor,
]
