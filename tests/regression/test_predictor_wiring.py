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


"""Every predictor input must be something the driver can actually supply.

A predictor consumes old state and driving forces -- nothing else. If it asks for
anything more, ``TransientDriver`` zero-fills it (deliberately: that is how initial
conditions work, mirroring C++ ``VariableStore::zero_undefined_input``) and the
predictor silently returns a degraded guess. Nothing fails: a predictor cannot
change the converged answer, so every gold still passes and the only symptom is
iteration count.

That has already produced two false negatives in this repository -- a predictor
asking for `deformation_rate` in a scenario that prescribes
`spatial_velocity_gradient`, and a reused block asking for `R` under a scenario
that renames its rotation matrix. Both measured as "the predictor does nothing"
rather than as an error.

The invariant is static: a predictor input must be a lagged variable (``~n``), an
input of the residual model (i.e. a driving force), or time.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from neml2 import load_input
from neml2.models.common.ImplicitUpdate import ImplicitUpdate

REGRESSION = Path(__file__).parent

# A top-level block: its body runs to the first closing ``[]`` at the same indent.
_BLOCK = re.compile(r"^  \[(\w+)\]\n((?:.*?\n)*?)  \[\]\n", re.M)


def _predictor_blocks(text: str) -> list[str]:
    """Names of the ``ImplicitUpdate`` blocks that configure a predictor."""
    return [
        m.group(1)
        for m in _BLOCK.finditer(text)
        if "type = ImplicitUpdate" in m.group(2) and "predictor = " in m.group(2)
    ]


def _scenarios() -> list[Path]:
    return sorted(p for p in REGRESSION.rglob("*.i") if _predictor_blocks(p.read_text()))


@pytest.mark.parametrize("input_file", _scenarios(), ids=lambda p: str(p.relative_to(REGRESSION)))
def test_predictor_inputs_are_suppliable(input_file: Path):
    factory = load_input(input_file)
    names = _predictor_blocks(input_file.read_text())
    assert names, f"no predictor found in {input_file}"
    for name in names:
        iu = factory.get_model(name)
        assert isinstance(iu, ImplicitUpdate)
        assert iu.predictor is not None
        givens = set(iu.system.model.input_spec)
        unsuppliable = [n for n in iu.predictor.input_spec if "~" not in n and n not in givens]
        assert not unsuppliable, (
            f"[{name}]'s predictor asks for {unsuppliable}, which is neither a lagged "
            f"variable nor an input of the residual model. The driver will zero-fill it "
            f"and the predictor will quietly return a worse guess."
        )
