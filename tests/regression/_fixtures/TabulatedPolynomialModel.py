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

"""Python-native port of the C++ test helper ``TabulatedPolynomialModel``.

Mirrors ``tests/src/TabulatedPolynomialModel.cxx``. Forward semantics:

    x = stack(s, T, s1, s2)                                  (..., 4)
    y_cell[i,j,k] = A0[i,j,k]
                  + sum_l A1[i,j,k,l] * x[l]
                  + sum_l A2[i,j,k,l] * x[l]^2               (..., 2, 3, 3)
    i = smooth_index(s, s_lb, s_ub)                          (..., 2)
    j = smooth_index(T, T_lb, T_ub)                          (..., 3)
    ij[i,j] = i[i] * j[j]                                    (..., 2, 3)
    y[k] = sum_{i,j} ij[i,j] * y_cell[i,j,k]                 (..., 3)

with ``smooth_index(x, lb, ub) = 0.5 (sigmoid(k(x-lb)) - sigmoid(k(x-ub)))``.

Test-fixture only. Self-registers via ``@register_neml2_object`` so the
``polynomial`` regression scenario can instantiate it from HIT.
"""

from __future__ import annotations

from typing import Any, cast

import nmhit
import torch

from neml2 import allow_autograd
from neml2._warnings import TORCH_JIT_PY314, ignore_warnings
from neml2.factory import _NativeInputFile, register_neml2_object
from neml2.models.chain_rule import ChainRuleDict
from neml2.models.model import Model
from neml2.schema import HitSchema, input, option, output
from neml2.types import Scalar


def _resolve_tensor(factory: _NativeInputFile, spec: str) -> torch.Tensor:
    """Resolve a HIT [Tensors] cross-reference to a raw torch.Tensor."""
    val = factory.get_tensor(spec)
    if isinstance(val, torch.Tensor):
        return val
    # TensorWrapper — unwrap.
    return val.data


@register_neml2_object("TabulatedPolynomialModel")
class TabulatedPolynomialModel(Model):
    """Native mirror of the C++ test helper.

    Schema matches the C++ ``expected_options``: 4 Scalar inputs, 3 Scalar
    outputs, 7 raw-tensor buffers resolved from ``[Tensors]`` cross-refs, and
    one float option for the smooth-index sharpness.
    """

    hit = HitSchema(
        input("von_mises_stress", Scalar, "von Mises stress", attr="_s_var"),
        input("temperature", Scalar, "temperature", attr="_T_var"),
        input("internal_state_1", Scalar, "internal state 1", attr="_s1_var"),
        input("internal_state_2", Scalar, "internal state 2", attr="_s2_var"),
        output(
            "equivalent_plastic_strain_rate",
            Scalar,
            "equivalent plastic strain rate",
            attr="_ep_dot_var",
        ),
        output("internal_state_1_rate", Scalar, "internal state 1 rate", attr="_s1_dot_var"),
        output("internal_state_2_rate", Scalar, "internal state 2 rate", attr="_s2_dot_var"),
        option("A0", str, "Constant term in the polynomial", attr="_A0_name"),
        option("A1", str, "Linear term in the polynomial", attr="_A1_name"),
        option("A2", str, "Quadratic term in the polynomial", attr="_A2_name"),
        option(
            "stress_tile_lower_bounds",
            str,
            "Lower bounds for stress tiles",
            attr="_s_lb_name",
        ),
        option(
            "stress_tile_upper_bounds",
            str,
            "Upper bounds for stress tiles",
            attr="_s_ub_name",
        ),
        option(
            "temperature_tile_lower_bounds",
            str,
            "Lower bounds for temperature tiles",
            attr="_T_lb_name",
        ),
        option(
            "temperature_tile_upper_bounds",
            str,
            "Upper bounds for temperature tiles",
            attr="_T_ub_name",
        ),
        option(
            "index_sharpness",
            float,
            "Sharpness of the smooth index function",
            default=1.0,
            attr="_k",
        ),
    )

    # Names of the [Tensors] entries that back each buffer; resolved in
    # __init__ via the factory.
    _A0_name: str
    _A1_name: str
    _A2_name: str
    _s_lb_name: str
    _s_ub_name: str
    _T_lb_name: str
    _T_ub_name: str
    _k: float

    # Class-level declarations narrow what ``nn.Module.__getattr__`` returns
    # so pyright sees ``self.A0`` etc. as ``Tensor`` rather than the union
    # ``Tensor | Module | Any``. ``from_hit`` registers them via
    # ``register_buffer``.
    A0: torch.Tensor
    A1: torch.Tensor
    A2: torch.Tensor
    s_lb: torch.Tensor
    s_ub: torch.Tensor
    T_lb: torch.Tensor
    T_ub: torch.Tensor

    # Resolved variable names (after HIT override -> attr).
    _s_var: str
    _T_var: str
    _s1_var: str
    _s2_var: str
    _ep_dot_var: str
    _s1_dot_var: str
    _s2_dot_var: str

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> Any:
        # Use the default Model.from_hit to populate schema-driven attrs,
        # then resolve the [Tensors] cross-references into raw buffers.
        obj = super().from_hit(node, factory)
        for attr_name in ("A0", "A1", "A2", "s_lb", "s_ub", "T_lb", "T_ub"):
            spec = getattr(obj, f"_{attr_name}_name")
            tensor = _resolve_tensor(factory, spec).to(dtype=torch.float64)
            obj.register_buffer(attr_name, tensor)
        return obj

    def _smooth_index(self, x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        x0 = x.unsqueeze(-1)
        return 0.5 * (torch.sigmoid(self._k * (x0 - lb)) - torch.sigmoid(self._k * (x0 - ub)))

    def _forward_raw(
        self,
        s: torch.Tensor,
        T: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw-tensor forward implementing the C++ ``set_value`` body.

        Inputs are broadcast and stacked into ``x`` of shape ``(..., 4)``; the
        polynomial pushes that through A0/A1/A2 and the smooth-index outer
        product produces a ``(..., 3)`` output that is split into 3 Scalars.
        """
        # Broadcast everyone to a common shape so stack(-1) works on a list.
        s, T, s1, s2 = torch.broadcast_tensors(s, T, s1, s2)
        # x has shape (..., 4) then we add base dims so matmul against
        # A1/A2 (..., 2, 3, 3, 4) produces (..., 2, 3, 3, 1).
        x = torch.stack([s, T, s1, s2], dim=-1)
        x_b = x.unsqueeze(-2).unsqueeze(-2).unsqueeze(-1)

        i = self._smooth_index(s, self.s_lb, self.s_ub)  # (..., 2)
        j = self._smooth_index(T, self.T_lb, self.T_ub)  # (..., 3)
        ij = torch.matmul(i.unsqueeze(-1), j.unsqueeze(-2))  # (..., 2, 3)

        # y_cell shape (..., 2, 3, 3)
        y_cell = (
            self.A0.unsqueeze(-1) + torch.matmul(self.A1, x_b) + torch.matmul(self.A2, x_b * x_b)
        )
        y_cell = y_cell.squeeze(-1)

        # Contract ij with y_cell over the (2, 3) tile axes: out shape (..., 3).
        # Equivalent to einsum("...ij,...ijk->...k", ij, y_cell) without einsum:
        # multiply with ij[..., None] then sum over the (i, j) pair.
        out = (ij.unsqueeze(-1) * y_cell).sum(dim=(-3, -2))
        return out[..., 0], out[..., 1], out[..., 2]

    def forward(  # type: ignore[override]
        self,
        s: Scalar,
        T: Scalar,
        s1: Scalar,
        s2: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        ep_dot_raw, s1_dot_raw, s2_dot_raw = self._forward_raw(s.data, T.data, s1.data, s2.data)
        ep_dot = Scalar(ep_dot_raw)
        s1_dot = Scalar(s1_dot_raw)
        s2_dot = Scalar(s2_dot_raw)

        if v is None:
            return ep_dot, s1_dot, s2_dot

        # D-062 chain rule. The forward is a smooth composition of polynomial,
        # matmul, and sigmoid -- all autograd-friendly. Rather than derive
        # closed-form Jacobian rows by hand for a 4-input x 3-output coupled
        # system, each action routes its pushforward through torch.func.jvp
        # under an allow_autograd window opened inside the action body
        # (the v-aware caller invokes actions outside this construction
        # frame). This is a test fixture; the production native package
        # does not depend on it.
        actions = self._make_actions(s.data, T.data, s1.data, s2.data)

        v_out: ChainRuleDict = {}
        for out_name, out_wrap, out_actions in zip(
            (self._ep_dot_var, self._s1_dot_var, self._s2_dot_var),
            (ep_dot, s1_dot, s2_dot),
            actions,
            strict=True,
        ):
            v_out.update(self.apply_chain_rule(v, out_name, out_actions, output=out_wrap))
        return ep_dot, s1_dot, s2_dot, v_out

    def _make_actions(
        self,
        s: torch.Tensor,
        T: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> tuple[dict, dict, dict]:
        """Build chain-rule action dicts for each of the three outputs.

        Each action takes an incoming Scalar tangent ``V`` (with leading seed
        axis K) for one input variable and returns the Scalar contribution to
        the named output. We use torch.func.jvp once per (output, input) pair
        with a primal-and-tangent batch already vectorised over K, so a single
        forward suffices per call site.
        """
        from torch.func import jvp

        input_vars = (self._s_var, self._T_var, self._s1_var, self._s2_var)
        primals = (s, T, s1, s2)

        def _make_action(out_idx: int, in_idx: int):
            def action(V):
                # V is a Scalar with leading seed axis K. Stage tangents for
                # the in_idx-th primal and zero tangents for the others.
                Vd = V.data
                # Broadcast Vd against the primal so the jvp tangent shares
                # the primal's shape. The primal is a static (no-K) tensor;
                # we need to lift it to (K, *primal.shape) by broadcasting
                # against Vd.
                K_primals = tuple(
                    p.clone() if p.shape == Vd.shape else p.expand_as(Vd).clone() for p in primals
                )
                # Tangents: zero everywhere except slot `in_idx`.
                tangents = [torch.zeros_like(K_primals[k]) for k in range(4)]
                tangents[in_idx] = Vd.expand_as(K_primals[in_idx])

                def fn(s_, T_, s1_, s2_):
                    return self._forward_raw(s_, T_, s1_, s2_)

                # The autograd guard is process-level: the v-aware caller
                # invokes this action *after* the construction-time
                # allow_autograd window has closed, so we must reopen it here
                # for each invocation. This is a test fixture (the production
                # native package doesn't depend on it) — the analytical
                # Jacobian for the 4-input x 3-output coupled polynomial +
                # sigmoid is a 12-entry expression that's not worth deriving
                # by hand.
                with (
                    allow_autograd(
                        "TabulatedPolynomialModel test fixture: routing the "
                        "pushforward through torch.func.jvp."
                    ),
                    # First forward-mode jvp triggers torch's one-time JVP-decomposition
                    # registration, which on Python 3.14 warns about torch's own
                    # deprecated torch.jit.script (neml2 never calls it).
                    ignore_warnings(TORCH_JIT_PY314),
                ):
                    # torch.func.jvp is typed as a union of 2-/3-tuples
                    # (the 3-tuple branch is the ``has_aux=True`` overload
                    # we don't use). Cast to narrow.
                    _, jvp_out = cast("tuple[Any, Any]", jvp(fn, K_primals, tuple(tangents)))
                # K_primals were lifted to Vd.shape so jvp_out shares Vd's
                # leading-K layout. Re-attach V's K metadata so the chain
                # rule downstream sees the K axis as K (not batch).
                return Scalar(
                    jvp_out[out_idx],
                    sub_batch_ndim=V.sub_batch_ndim,
                    sub_batch_state=V.sub_batch_state,
                    sub_batch_meta=V.sub_batch_meta,
                    k_ndim=V.k_ndim,
                    k_state=V.k_state,
                    k_pairing=V.k_pairing,
                )

            return action

        actions = []
        for out_idx in range(3):
            actions.append({iv: _make_action(out_idx, i) for i, iv in enumerate(input_vars)})
        return tuple(actions)


__all__ = ["TabulatedPolynomialModel"]
