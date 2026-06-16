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

"""Verify every code cell in the given notebooks has been executed.

Used by the pre-commit hook to keep committed ``.ipynb`` files in a
fully-executed state. Sphinx's myst-nb runs with ``nb_execution_mode =
"off"`` so the docs render the persisted outputs verbatim -- an
unexecuted cell would render with an empty output region (or, worse, a
stale output from a previous run if the source has since changed).

A code cell is considered executed when its ``execution_count`` is not
``None``. This mirrors v2's check at ``doc/scripts/examples.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat


def _unexecuted_cells(nb_path: Path) -> list[int]:
    nb = nbformat.read(nb_path, as_version=4)
    return [
        i
        for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and cell.execution_count is None
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()

    failed: list[tuple[Path, list[int]]] = []
    for nb_path in args.notebooks:
        unexecuted = _unexecuted_cells(nb_path)
        if unexecuted:
            failed.append((nb_path, unexecuted))

    if failed:
        for nb_path, indices in failed:
            print(
                f"{nb_path}: {len(indices)} unexecuted code cell(s) at index {indices}",
                file=sys.stderr,
            )
        print(
            "Re-execute the notebook(s) end-to-end and commit the updated outputs.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
