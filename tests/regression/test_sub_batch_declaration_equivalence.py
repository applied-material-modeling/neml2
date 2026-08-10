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

"""Declaring a sub-batch extent is equivalent to hand-shaping the IC that used
to imply it.

``per_slip_hardening`` establishes its 12-slip-system axis with an initial
condition built as a ``(20, 12)`` tensor carrying ``sub_batch_ndim=1`` -- the
initial condition is the only thing in the file that knows the extent, so it
ends up declaring it by accident. ``per_slip_hardening_declared`` is the same
scenario with the extent stated in ``[Settings]/example_batch_shape`` and the
initial condition reduced to the value it always was.

Each scenario pins itself against its own gold in the regression sweep. What
that cannot say -- because the two golds are separate files -- is that the
change of spelling did not change the answer. This does.
"""

from __future__ import annotations

from pathlib import Path

import torch

from neml2 import load_input

_CP = Path(__file__).parent / "solid_mechanics/crystal_plasticity"
_HAND_SHAPED = _CP / "per_slip_hardening/model.i"
_DECLARED = _CP / "per_slip_hardening_declared/model.i"

#: The two entries that legitimately differ: the initial condition's own record
#: and the step-1 history copied straight from it. In the declared scenario
#: those really are a scalar, because that is the point -- the value no longer
#: carries the shape. Every other one of the ~1100 recorded fields, including
#: every later step of the per-slip unknown, must agree exactly.
_IC_RECORDS = frozenset({"output.0.dislocation_density", "input.1.dislocation_density~1"})


def _run(path: Path) -> dict[str, torch.Tensor]:
    driver = load_input(path).get_driver("driver")
    driver.run()
    return {k: v.detach() for k, v in driver.result().items()}


def test_declaring_the_sub_batch_extent_reproduces_the_hand_shaped_ic():
    hand_shaped = _run(_HAND_SHAPED)
    declared = _run(_DECLARED)

    assert set(hand_shaped) == set(declared)

    for key, expected in hand_shaped.items():
        if key in _IC_RECORDS:
            # Same value, differently shaped: a scalar seed vs the (20, 12)
            # tensor it broadcasts into.
            torch.testing.assert_close(declared[key].expand_as(expected), expected, rtol=0, atol=0)
            continue
        torch.testing.assert_close(
            declared[key], expected, rtol=0, atol=0, msg=lambda m, k=key: f"{k}: {m}"
        )
