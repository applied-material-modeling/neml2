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

import importlib
import sys
from pathlib import Path

import neml2.models as native_models


def test_model_packages_import_all_leaf_files():
    models_root = Path(native_models.__file__).parent
    for init_file in models_root.rglob("__init__.py"):
        package_dir = init_file.parent
        leaf_files = [
            path
            for path in package_dir.glob("*.py")
            if path.name != "__init__.py" and not path.name.startswith("_")
        ]
        if not leaf_files:
            continue

        package_name = "neml2.models." + package_dir.relative_to(models_root).as_posix().replace(
            "/", "."
        )
        importlib.import_module(package_name)
        missing = [
            path.stem for path in leaf_files if f"{package_name}.{path.stem}" not in sys.modules
        ]
        assert missing == []
