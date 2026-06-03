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

"""Single driver that runs every native ``Verification`` input file.

Walks every ``.i`` file under this directory, locates its ``[Drivers]``
``type = Verification`` block, and calls ``.run()``. Pass ↔ all reference
variables match the native output within the file's declared ``rtol`` /
``atol``.

To add a verification scenario, drop ``<scenario>/<name>.i`` plus its CSV
reference (``<stem>.csv`` or ``reference.csv``) under
``python/tests/native/verification/<domain>/<scenario>/`` — no Python needed.
"""

from __future__ import annotations

from pathlib import Path

import nmhit
import pytest

from neml2 import load_input
from neml2.drivers import Verification

_VERIFICATION_DIR = Path(__file__).parent
_INPUTS = sorted(_VERIFICATION_DIR.rglob("*.i"))


def _find_verification_driver_name(input_file: Path) -> str:
    """Return the name of the ``[Drivers]`` sub-block whose type is ``Verification``."""
    root = nmhit.parse_file(input_file, [], [])
    for top in root.children(nmhit.NodeType.Section):
        if top.path() != "Drivers":
            continue
        for child in top.children(nmhit.NodeType.Section):
            if child.param_optional_str("type", "") == "Verification":
                return child.path().rsplit("/", 1)[-1]
    raise ValueError(f"{input_file} has no [Drivers] block of type 'Verification'")


@pytest.mark.parametrize(
    "input_file",
    _INPUTS,
    ids=[str(p.relative_to(_VERIFICATION_DIR)) for p in _INPUTS],
)
def test_verification(input_file: Path):
    name = _find_verification_driver_name(input_file)
    factory = load_input(input_file)
    driver = factory.get_driver(name)
    assert isinstance(driver, Verification)
    assert driver.run()
