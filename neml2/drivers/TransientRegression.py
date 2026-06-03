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

"""Native ``TransientRegression`` — mirror of C++ ``tests/src/TransientRegression.cxx``.

Wraps a paired :class:`TransientDriver`, runs it, then diffs the in-memory
result dict against a TorchScript ``gold/result.pt`` produced by the C++
regression suite. ``torch.jit.load`` returns the gold as a
``RecursiveScriptModule`` whose ``named_buffers()`` enumerate the original
``input.<step>.<var>`` / ``output.<step>.<var>`` layout.

The diff is asymmetric:
- Every gold key MUST exist in the native result and match within tolerance.
- Extra keys in the native result (e.g. inputs the native model tracks that
  the C++ side never recorded, like ``t~1`` for some time integrators) are
  IGNORED — this is a cross-backend parity check, not a strict
  identity check on the result shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nmhit
import torch

from ..driver import Driver
from ..factory import register_native
from .TransientDriver import TransientDriver

if TYPE_CHECKING:
    from ..factory import _NativeInputFile


@register_native("TransientRegression")
class TransientRegression(Driver):
    """Run a TransientDriver and diff its result against a gold ``.pt`` file."""

    def __init__(
        self,
        driver: TransientDriver,
        reference: Path,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
    ) -> None:
        self.driver = driver
        self.reference = reference
        self.rtol = rtol
        self.atol = atol

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> TransientRegression:
        driver_name = node.param_str("driver")
        driver = factory.get_driver(driver_name)
        if not isinstance(driver, TransientDriver):
            raise TypeError(
                f"TransientRegression expects a TransientDriver in 'driver', got "
                f"{type(driver).__name__}"
            )

        reference_str = node.param_str("reference")
        reference = Path(reference_str)
        if not reference.is_absolute():
            # Resolve relative to the .i file's directory, matching the C++ side.
            reference = factory._path.resolve().parent / reference

        rtol = node.param_optional_float("rtol", 1.0e-5)
        atol = node.param_optional_float("atol", 1.0e-8)

        return cls(driver=driver, reference=reference, rtol=rtol, atol=atol)

    def run(self) -> bool:
        self.driver.run()
        result = self.driver.result()

        if not self.reference.exists():
            raise FileNotFoundError(f"TransientRegression reference not found: {self.reference}")
        gold_module = torch.jit.load(str(self.reference))
        # Use ``state_dict`` (NOT ``named_buffers``): the C++ TransientDriver's
        # ``advance_step`` does a shallow Tensor copy from ``result_out[k]``
        # into ``result_in[k+1]``, so e.g. ``output.<k>.flow_rate`` and
        # ``input.<k+1>.flow_rate~1`` share storage. ``named_buffers`` (default
        # ``remove_duplicate=True``) drops one of every such pair, hiding
        # roughly a third of the gold entries from the comparison.
        gold = dict(gold_module.state_dict())

        # Pass 1 — every gold key must exist in the result.
        missing = sorted(k for k in gold if k not in result)
        if missing:
            raise AssertionError(
                f"TransientRegression: gold has {len(missing)} key(s) absent from "
                f"native result; first few: {missing[:5]}\n"
                f"  reference: {self.reference}"
            )

        # Pass 2 — each gold key matches within tolerance.
        # detach() so requires_grad differences between the native autograd
        # path and the C++ gold (which is always grad-free) don't trip
        # assert_close's tensor-property checks.
        mismatches: list[str] = []
        for key, ref in gold.items():
            res = result[key].detach()
            try:
                torch.testing.assert_close(res, ref, rtol=self.rtol, atol=self.atol)
            except AssertionError:
                # Trim torch's multi-line msg to one short summary line.
                diff = (res - ref).abs()
                max_abs = float(diff.max())
                rel_denom = ref.abs().clamp(min=torch.finfo(ref.dtype).eps)
                max_rel = float((diff / rel_denom).max())
                mismatches.append(f"  {key}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
                if len(mismatches) >= 10:  # cap output noise
                    mismatches.append("  ... (further mismatches suppressed)")
                    break
        if mismatches:
            raise AssertionError(
                f"TransientRegression vs {self.reference} failed "
                f"(rtol={self.rtol}, atol={self.atol}):\n" + "\n".join(mismatches)
            )

        return True


__all__ = ["TransientRegression"]
