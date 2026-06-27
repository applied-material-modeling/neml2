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
result dict against a gold ``.pt`` reference. The reader (:func:`_load_gold`)
handles both the new plain-``torch.save``-dict format that the Python
:meth:`TransientDriver.save_result` now emits and the legacy TorchScript-
module format produced by the retired C++ regression pipeline (most existing
on-disk goldens are still in the legacy format).

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

from ..factory import register_neml2_object
from ..schema import HitSchema, dependency, option
from .driver import Driver
from .TransientDriver import TransientDriver

if TYPE_CHECKING:
    from ..factory import _NativeInputFile


def _load_gold(path: Path) -> dict[str, torch.Tensor]:
    """Load a gold-file as a flat ``{key: tensor}`` dict.

    Supports two on-disk formats:

    - **New** — a plain ``torch.save({...})`` dict written by
      :meth:`TransientDriver.save_result`. Loaded via
      ``torch.load(..., weights_only=True)``.
    - **Legacy** — a TorchScript ``nn.Module`` of registered buffers
      produced by the retired C++ pipeline (and by older Python
      ``save_result`` writes). Loaded via ``torch.jit.load`` and flattened
      via ``state_dict()`` — NOT ``named_buffers`` (the C++ driver shares
      storage between ``output.<k>.X`` and ``input.<k+1>.X~1``, and
      ``named_buffers`` with the default ``remove_duplicate=True`` drops
      one of every such pair).

    Detection is by sniff: try the plain-dict path first, fall back to
    TorchScript on any failure. Most existing on-disk goldens are still
    legacy until they're regenerated.
    """
    try:
        payload = torch.load(str(path), weights_only=True)
    except Exception:  # noqa: BLE001 — torch.load raises a zoo of types on JIT archives
        return dict(torch.jit.load(str(path)).state_dict())
    if isinstance(payload, dict):
        return {str(k): v for k, v in payload.items()}
    # Unexpected: file loaded as something else (e.g. a bare Tensor). Treat as
    # legacy and re-route through torch.jit.
    return dict(torch.jit.load(str(path)).state_dict())


@register_neml2_object("TransientRegression")
class TransientRegression(Driver):
    """Run a TransientDriver and diff its result against a gold ``.pt`` file."""

    hit = HitSchema(
        dependency("driver", "get_driver", "The TransientDriver to run before diffing."),
        option(
            "reference",
            str,
            "Path to the gold ``.pt`` file. Resolved relative to the input file's directory "
            "when not absolute.",
        ),
        option(
            "rtol",
            float,
            "Relative tolerance for per-tensor comparison.",
            default=1.0e-5,
        ),
        option(
            "atol",
            float,
            "Absolute tolerance for per-tensor comparison.",
            default=1.0e-8,
        ),
    )

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
        gold = _load_gold(self.reference)

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
