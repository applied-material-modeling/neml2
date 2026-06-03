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

"""Python-native port of the C++ test helper ``TorchScriptFlowRate``.

Mirrors ``tests/src/TorchScriptFlowRate.cxx``. The forward loads a
TorchScript module from disk at construction and, at evaluation time,
calls ``surrogate(s, T, G, C)`` with ``G = 0.1`` / ``C = 0.2`` constants
(matching the C++ hard-coded scalar fills). The TorchScript surrogate
returns a single scalar ``ep_dot``.

Test-fixture only. Self-registers via ``@register_native``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nmhit
import torch

from neml2 import allow_autograd
from neml2.chain_rule import ChainRuleDict
from neml2.factory import _NativeInputFile, register_native
from neml2.model import Model
from neml2.schema import HitSchema, input, option, output
from neml2.types import Scalar


@register_native("TorchScriptFlowRate")
class TorchScriptFlowRate(Model):
    """Native mirror of the C++ test helper.

    Schema matches the C++ ``expected_options``: 2 Scalar inputs (von Mises
    stress, temperature), 1 Scalar output (equivalent plastic strain rate),
    and a ``torch_script`` option pointing to the ``.pt`` file (resolved
    relative to the input file's directory, mirroring how
    ``TransientRegression.reference`` is handled).
    """

    hit = HitSchema(
        input("von_mises_stress", Scalar, "The von Mises stress", attr="_s_var"),
        input("temperature", Scalar, "The temperature", attr="_T_var"),
        output(
            "equivalent_plastic_strain_rate",
            Scalar,
            "The equivalent plastic strain rate",
            attr="_ep_dot_var",
        ),
        option(
            "torch_script",
            str,
            "Path to the TorchScript module backing the forward operator",
            attr="_torch_script_path",
        ),
    )

    _s_var: str
    _T_var: str
    _ep_dot_var: str
    _torch_script_path: str

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> Any:
        obj = super().from_hit(node, factory)
        path = Path(obj._torch_script_path)
        if not path.is_absolute():
            path = factory._path.resolve().parent / path
        obj._surrogate = torch.jit.load(str(path))
        return obj

    def _forward_raw(self, s: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        # Hard-coded grain-size / stoichiometry constants matching the C++
        # ``Scalar::full(0.1, ...)`` / ``Scalar::full(0.2, ...)`` calls.
        G = torch.full_like(s, 0.1)
        C = torch.full_like(s, 0.2)
        # The surrogate returns a single tensor (Scalar in NEML2 terms).
        return self._surrogate(s, T, G, C)

    def forward(  # type: ignore[override]
        self,
        s: Scalar,
        T: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        out_data = self._forward_raw(s.data, T.data)
        out = Scalar(out_data)
        if v is None:
            return out

        # D-062 chain rule. TorchScript modules are opaque so closed-form
        # Jacobians are out of reach; route through torch.func.jvp. As with
        # TabulatedPolynomialModel this is acceptable for a test fixture that
        # never participates in AOTI export. The autograd guard is process-
        # level: the v-aware caller invokes actions outside this construction
        # frame, so each action reopens its own allow_autograd window.
        from torch.func import jvp

        s_data = s.data
        T_data = T.data

        def _make_action(in_idx: int):
            def action(V):
                Vd = V.data
                primals_full = (s_data, T_data)
                primals = tuple(
                    p.expand_as(Vd).clone() if p.shape != Vd.shape else p for p in primals_full
                )
                tangents = [torch.zeros_like(primals[k]) for k in range(2)]
                tangents[in_idx] = Vd.expand_as(primals[in_idx])
                with allow_autograd(
                    "TorchScriptFlowRate test fixture: pushforward via torch.func.jvp."
                ):
                    _, jvp_out = jvp(self._forward_raw, primals, tuple(tangents))
                return Scalar(jvp_out)

            return action

        actions = {
            self._s_var: _make_action(0),
            self._T_var: _make_action(1),
        }

        return out, self.apply_chain_rule(v, self._ep_dot_var, actions, output=out)


__all__ = ["TorchScriptFlowRate"]
