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

"""Python-native mirror of C++ ``common/ConstantExtrapolationPredictor.h``."""

from __future__ import annotations

from typing import cast

import torch

from ...factory import register_neml2_object
from ...schema import HitSchema, option
from ...types import (
    MRP,
    R2,
    SR2,
    Scalar,
    TensorWrapper,
    gt,
    where,
)
from ...types.functions import abs as wrap_abs
from .._hit import _opt_list_str
from ..chain_rule import ChainRuleDict
from ..model import Model


def _read_list_str(node, name):  # noqa: ANN001, ANN202
    return list(node.param_list_str(name))


@register_neml2_object("ConstantExtrapolationPredictor")
class ConstantExtrapolationPredictor(Model):
    """Initial guess for an implicit update: each unknown takes its ``~1`` value.

    For an ``ImplicitUpdate`` with unknowns ``{u_i}``, this predictor reads
    ``u_i~1`` and outputs $u_i$. Used by ``ImplicitUpdate`` to seed Newton.
    Matches the C++ ``ConstantExtrapolationPredictor`` HIT signature.
    """

    hit = HitSchema(
        option(
            "unknowns_SR2",
            list,
            "The unknowns to extrapolate of type SR2",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_Scalar",
            list,
            "The unknowns to extrapolate of type Scalar",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_MRP",
            list,
            "The unknowns to extrapolate of type MRP",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_R2",
            list,
            "The unknowns to extrapolate of type R2",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "cold",
            list,
            "Optional `unknown:variable` pairs naming a COLD value per unknown. "
            "On the step where there is no history -- the first -- that unknown "
            "is seeded from the named variable instead of from `u~1`, which lets "
            "a physics-based predictor supply the cold start while this model "
            "keeps every later step. Name only the unknowns you want seeded; the "
            "rest fall back to `u~1` exactly as before. The type comes from "
            "whichever `unknowns_*` list the name appears in, so there is no "
            "second list to keep in step.",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "time",
            str,
            "Time. Only consumed when a cold mapping is given -- the "
            "no-history test needs the time history.",
            default="t",
            reader=lambda node, name: node.param_str(name),
            optional_reader=lambda node, name, default: (
                node.param_str(name) if node.find(name) is not None else default
            ),
        ),
    )

    def __init__(
        self,
        unknowns_SR2: list[str],
        unknowns_Scalar: list[str],
        unknowns_MRP: list[str] | None = None,
        unknowns_R2: list[str] | None = None,
        cold: list[str] | None = None,
        time: str = "t",
    ) -> None:
        super().__init__()
        self._sr2 = list(unknowns_SR2)
        self._scalar = list(unknowns_Scalar)
        self._rot = list(unknowns_MRP or [])
        self._r2 = list(unknowns_R2 or [])
        self.input_spec = {
            **{f"{u}~1": SR2 for u in self._sr2},
            **{f"{u}~1": Scalar for u in self._scalar},
            **{f"{u}~1": MRP for u in self._rot},
            **{f"{u}~1": R2 for u in self._r2},
        }
        self.output_spec = {
            **{u: SR2 for u in self._sr2},
            **{u: Scalar for u in self._scalar},
            **{u: MRP for u in self._rot},
            **{u: R2 for u in self._r2},
        }

        # Optional cold mapping, as `unknown:variable` pairs. Explicit rather
        # than positional so it is partial by construction -- name only the
        # unknowns you want seeded -- and so it cannot drift against
        # `unknowns_*`: an unknown that does not exist is an error, not a
        # silently-ignored entry.
        types: dict[str, type[TensorWrapper]] = {
            **{u: SR2 for u in self._sr2},
            **{u: Scalar for u in self._scalar},
            **{u: MRP for u in self._rot},
            **{u: R2 for u in self._r2},
        }
        self._cold: dict[str, str] = {}
        for entry in cold or []:
            name, sep, value = entry.partition(":")
            if not sep or not name or not value:
                raise ValueError(
                    f"{type(self).__name__}: cold entry {entry!r} is not `unknown:variable`."
                )
            if name not in types:
                raise ValueError(
                    f"{type(self).__name__}: cold entry {entry!r} names {name!r}, "
                    f"which is not one of the unknowns {sorted(types)}."
                )
            if name in self._cold:
                raise ValueError(
                    f"{type(self).__name__}: unknown {name!r} is given a cold value twice."
                )
            self._cold[name] = value
        self._time = time
        if self._cold:
            # The no-history test reads the time history; only pay for those
            # inputs when a cold mapping is actually in play, so an existing
            # input file's call signature is untouched.
            self.input_spec = {
                **self.input_spec,
                time: Scalar,
                f"{time}~1": Scalar,
                f"{time}~2": Scalar,
                **{self._cold[u]: types[u] for u in self.output_spec if u in self._cold},
            }

    def forward(  # type: ignore[override]
        self,
        *inputs: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        if not self._cold:
            # Unchanged path: a pure pass-through u_i~1 -> u_i, identity gradient.
            outs: tuple[TensorWrapper, ...] = tuple(inputs)
        else:
            n = len(self.output_spec)
            hist = list(inputs[:n])
            t_val, t_n, t_nm1 = (cast(Scalar, x) for x in inputs[n : n + 3])
            cold_vals = list(inputs[n + 3 :])
            # Same no-history test the linear predictor uses: with no second
            # history point there is nothing to have converged from, which is a
            # sharper question than "is this variable small" -- a quantity with a
            # non-zero initial value (an identity Fp, a physical dislocation
            # density) is not small on the step where it is nonetheless cold.
            eps = torch.finfo(t_val.dtype).eps
            warm = gt(wrap_abs(t_n - t_nm1), eps)
            it = iter(cold_vals)
            outs = tuple(
                where(warm, h, next(it)) if u in self._cold else h
                for h, u in zip(hist, self.output_spec, strict=True)
            )
        if v is None:
            # Match the leaf-Model convention: single output unwrapped to a
            # bare wrapper, multiple outputs as a tuple. Written this way (vs
            # a single ternary) so pyright correctly narrows `outs` past the
            # length check.
            if len(outs) == 1:
                return outs[0]
            return outs
        # Build v_out that simply renames each var~1 -> var. With a cold mapping
        # the history term survives only on the warm branch, and the cold value
        # carries the other one; a predictor is never differentiated in
        # practice, but the two branches are kept honest rather than dropped.
        v_out: ChainRuleDict = {}
        hist_names = list(self.input_spec)[: len(self.output_spec)]
        for hname, out in zip(hist_names, self.output_spec, strict=True):
            v_out[out] = dict(v.get(hname, {}))
            cold = self._cold.get(out)
            if cold:
                for leaf, blk in v.get(cold, {}).items():
                    v_out[out][leaf] = blk if leaf not in v_out[out] else v_out[out][leaf] + blk
        return (*outs, v_out)


__all__ = ["ConstantExtrapolationPredictor"]
