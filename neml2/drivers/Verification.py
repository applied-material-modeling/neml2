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

"""Native ``Verification`` driver — mirror of C++ ``VTestVerification``.

Wraps a :class:`TransientDriver`, runs it, then diffs the in-memory result
against per-variable reference tensors loaded from ``[Tensors]`` entries
(typically a ``CSV<Type>`` block).

The HIT options mirror C++ ``VTestVerification`` 1:1 with one rename: the
driver type is plain ``Verification`` (was ``VTestVerification``) because
references are loaded via the generic CSV path and the driver doesn't care
about the ``.vtest`` text format any more.

```
[verification]
  type = Verification
  driver = 'driver'
  SR2_names = 'output.stress'
  SR2_values = 'stresses'
  Scalar_names = 'output.equivalent_plastic_strain'
  Scalar_values = 'eqp_ref'
  rtol = 1e-5
  atol = 1e-8
  time_steps = '499'      # optional; if absent, compare every step
[]
```
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

import nmhit
import torch

from ..driver import Driver
from ..factory import register_neml2_object
from ..schema import HitField, HitSchema, dependency, option
from ..types import SR2, WR2, Scalar, Vec
from .TransientDriver import TransientDriver

if TYPE_CHECKING:
    from ..factory import _NativeInputFile
    from ..types._base import TensorWrapper


# Wrappers exposed at the HIT layer. Each entry is the type name a user
# writes in their input file. The values are the wrapper classes the
# verification driver knows how to compare. To extend, add an entry here.
_SUPPORTED_TYPES: dict[str, type[TensorWrapper]] = {
    "Scalar": Scalar,
    "SR2": SR2,
    "Vec": Vec,
    "WR2": WR2,
}


def _read_token_list(node: nmhit.Node, name: str) -> list[str]:
    """Read a whitespace-separated HIT param as a list; return [] if empty/missing."""
    raw = node.param_optional_str(name, "")
    return raw.split() if raw else []


def _typed_ref_fields() -> tuple[HitField, ...]:
    """Build the ``<Type>_names`` / ``<Type>_values`` option pairs the driver
    consumes for every wrapper class in :data:`_SUPPORTED_TYPES`.

    Documentation-only — :meth:`Verification.from_hit` parses them directly. The
    schema exists so ``neml2-syntax`` can render the full HIT surface.
    """
    fields: list[HitField] = []
    for type_name in _SUPPORTED_TYPES:
        fields.append(
            option(
                f"{type_name}_names",
                list,
                f"Result-buffer variable names (e.g. ``output.stress``) whose per-step "
                f"values are compared against ``{type_name}_values``.",
                default=[],
            )
        )
        fields.append(
            option(
                f"{type_name}_values",
                list,
                f"[Tensors] block names producing the reference {type_name} per result "
                f"variable named in ``{type_name}_names``; same length and order.",
                default=[],
            )
        )
    return tuple(fields)


@register_neml2_object("Verification")
class Verification(Driver):
    """Run a TransientDriver and diff its result against per-variable references."""

    hit = HitSchema(
        dependency("driver", "get_driver", "The TransientDriver to run before diffing."),
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
        option(
            "time_steps",
            list,
            "Optional whitespace-separated list of step indices to compare. When absent, "
            "every step is checked. A single step entry switches to snapshot comparison "
            "(the reference is treated as the value at that step, not time-axis-sliced).",
            default=[],
        ),
        *_typed_ref_fields(),
    )

    def __init__(
        self,
        driver: TransientDriver,
        refs: dict[str, TensorWrapper],
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
        time_steps: tuple[int, ...] = (),
    ) -> None:
        self.driver = driver
        self.refs = refs
        self.rtol = rtol
        self.atol = atol
        self.time_steps = tuple(sorted(time_steps))

    # ── HIT parsing ───────────────────────────────────────────────────────────

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> Verification:
        driver_name = node.param_str("driver")
        driver = factory.get_driver(driver_name)
        if not isinstance(driver, TransientDriver):
            raise TypeError(
                f"Verification expects a TransientDriver in 'driver', got {type(driver).__name__}"
            )

        # Walk each supported tensor type and pair up ``<T>_names`` with
        # ``<T>_values``. Both sides must have the same length per type; the
        # name strings become result-buffer prefixes (e.g. ``output.stress``).
        refs: dict[str, TensorWrapper] = {}
        for type_name, wrapper_cls in _SUPPORTED_TYPES.items():
            names = _read_token_list(node, f"{type_name}_names")
            values = _read_token_list(node, f"{type_name}_values")
            if len(names) != len(values):
                raise ValueError(
                    f"Verification: {type_name}_names has {len(names)} entries but "
                    f"{type_name}_values has {len(values)}"
                )
            for var_name, ref_tensor_name in zip(names, values, strict=True):
                ref = factory.get_tensor(ref_tensor_name)
                if not isinstance(ref, wrapper_cls):
                    raise TypeError(
                        f"Verification: reference {ref_tensor_name!r} for "
                        f"{type_name}_names entry {var_name!r} must be {wrapper_cls.__name__}, "
                        f"got {type(ref).__name__}"
                    )
                if var_name in refs:
                    raise ValueError(
                        f"Verification: duplicate reference variable name {var_name!r}"
                    )
                refs[var_name] = ref

        if not refs:
            raise ValueError(
                "Verification: no reference variables provided (set "
                "<T>_names/<T>_values for at least one supported type)"
            )

        rtol = node.param_optional_float("rtol", 1.0e-5)
        atol = node.param_optional_float("atol", 1.0e-8)

        time_steps_raw = _read_token_list(node, "time_steps")
        time_steps = tuple(int(x) for x in time_steps_raw)

        return cls(
            driver=driver,
            refs=refs,
            rtol=rtol,
            atol=atol,
            time_steps=time_steps,
        )

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> bool:
        self.driver.run()
        result = self.driver.result()

        mismatches: list[str] = []
        for ref_name, ref_wrapper in self.refs.items():
            self._compare_one(ref_name, ref_wrapper, result, mismatches)
            if len(mismatches) >= 10:
                mismatches.append("  ... (further mismatches suppressed)")
                break

        if mismatches:
            raise AssertionError(
                f"Verification failed (rtol={self.rtol}, atol={self.atol}):\n"
                + "\n".join(mismatches)
            )
        return True

    def _compare_one(
        self,
        ref_name: str,
        ref_wrapper: TensorWrapper,
        result: dict[str, torch.Tensor],
        mismatches: list[str],
    ) -> None:
        """Compare every per-step result buffer matching *ref_name* against *ref_wrapper*.

        ``ref_name`` is the bare variable identifier (e.g. ``output.stress``).
        Result keys carry an extra step token between prefix and suffix (e.g.
        ``output.0.stress``, ``output.1.stress``, ...). We expand the bare
        name into per-step keys, then compare each against the corresponding
        row of the reference (or the whole reference when ``time_steps`` is
        a single value — the snapshot-comparison quirk inherited from C++).
        """
        tokens = ref_name.split(".")
        if len(tokens) < 2:
            mismatches.append(f"  invalid reference variable name {ref_name!r}")
            return

        prefix, suffix = tokens[0], tokens[1:]

        # Discover how many steps the result has for this variable.
        nstep = 0
        found_any = False
        for resname in result:
            rest = resname.split(".")
            if len(rest) != len(tokens) + 1:
                continue
            if rest[0] != prefix:
                continue
            if rest[2:] != suffix:
                continue
            try:
                step = int(rest[1])
            except ValueError:
                continue
            nstep = max(nstep, step + 1)
            found_any = True

        if not found_any:
            mismatches.append(f"  result is missing variable {ref_name!r}")
            return

        ref_data = ref_wrapper.data
        single_snapshot = len(self.time_steps) == 1

        j = 0
        for i in range(nstep):
            if self.time_steps and not _binary_search(self.time_steps, i):
                continue
            if single_snapshot:
                # Reference is the snapshot at the only selected time step —
                # not time-axis-sliced. Mirrors VTestVerification.cxx:182.
                refi = ref_data
            else:
                refi = ref_data[j].squeeze()
            j += 1

            key = ".".join([prefix, str(i), *suffix])
            if key not in result:
                # Allow zero references at missing keys (C++ behavior); error on
                # any non-zero magnitude.
                if not torch.allclose(refi, torch.zeros_like(refi)):
                    mismatches.append(f"  result is missing variable {key!r}")
                continue

            resi = result[key].detach().squeeze()
            try:
                torch.testing.assert_close(resi, refi, rtol=self.rtol, atol=self.atol)
            except AssertionError:
                diff = (resi - refi).abs()
                max_abs = float(diff.max())
                rel_denom = refi.abs().clamp(min=torch.finfo(refi.dtype).eps)
                max_rel = float((diff / rel_denom).max())
                mismatches.append(f"  {key}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
                if len(mismatches) >= 10:
                    return


def _binary_search(sorted_seq: tuple[int, ...], target: int) -> bool:
    """Return True if *target* is in *sorted_seq* (which must be ascending)."""
    idx = bisect.bisect_left(sorted_seq, target)
    return idx < len(sorted_seq) and sorted_seq[idx] == target


__all__ = ["Verification"]
