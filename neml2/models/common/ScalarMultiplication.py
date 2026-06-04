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

"""Python-native mirror of C++ ``common/ScalarMultiplication.h``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...chain_rule import ChainRuleAction, ChainRuleDict, SecondOrderChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, option, output, parameter, var_inputs
from ...types import Scalar

if TYPE_CHECKING:
    import nmhit


def _read_list_bool(node: nmhit.Node, name: str) -> list[bool]:
    return list(node.param_list_bool(name))


def _opt_list_bool(node: nmhit.Node, name: str, default: list[bool]) -> list[bool]:
    """nmhit lacks param_optional_list_bool; emulate via find + list read."""
    return list(node.param_list_bool(name)) if node.find(name) is not None else list(default)


@register_neml2_object("ScalarMultiplication")
class ScalarMultiplication(Model):
    """Calculate the product of multiple Scalar variables with a constant scaling
    coefficient. Using reciprocal, one can have the reciprocity of each variable
    """

    # Multilinear in the from-variables (each appears to the first power, or
    # to the -1 power if reciprocal). Smooth in the smooth-everywhere region;
    # second-order chain-rule cross-terms expressed as primal-shape bilinears
    # (the framework iterates the seed-pair outer).
    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        var_inputs("from", Scalar, "Scalar variables to be multiplied", attr="_from_vars"),
        output("to", Scalar, "The multiplicative product", attr="_to"),
        parameter(
            "scaling",
            Scalar,
            "The scaling coefficient to multiply to the final product",
            default="1",
            attr="scaling",
            allow_nonlinear=True,
        ),
        option(
            "reciprocal",
            list,
            "List of boolens, one for each variable, in which the reciprocity of the "
            "corresponding variable is taken. When the length of this list is 1, the same "
            "reciprocal condition applies to all variables.",
            default=[False],
            reader=_read_list_bool,
            optional_reader=_opt_list_bool,
            attr="_inv",
        ),
    )

    _from_vars: list[str]
    _to: str
    _inv: list[bool]
    scaling: Scalar

    def __post_init__(self) -> None:
        # Mirror the C++ neml_assert: reciprocal must be length 1 (broadcast)
        # or one entry per from-variable.
        n = len(self._from_vars)
        if len(self._inv) != 1 and len(self._inv) != n:
            raise ValueError(
                f"{type(self).__name__}: expected 1 or {n} entries in reciprocal, "
                f"got {len(self._inv)}."
            )
        # Expand length-1 to match the from-list, matching C++ ScalarMultiplication.
        if len(self._inv) == 1:
            self._inv = [self._inv[0]] * n

    def forward(  # type: ignore[override]
        self,
        *args: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # Split positional inputs: the leading structural inputs (one per
        # from-var) followed by the *nl_params pack holding any mode-3/4
        # promoted parameters (e.g. scaling promoted to a runtime input).
        n = len(self._from_vars)
        inputs, nl_params = args[:n], args[n:]
        if len(inputs) != n:
            raise ValueError(f"{type(self).__name__} expected {n} inputs, got {len(inputs)}")
        A = self._get_param("scaling", nl_params, Scalar)

        # Forward: r = A * prod_i ( x_i if not inv[i] else 1/x_i ).
        # Build as r = A * f_0 where f_i = x_i or 1/x_i, then accumulate.
        result = A / inputs[0] if self._inv[0] else A * inputs[0]
        for i in range(1, n):
            result = result / inputs[i] if self._inv[i] else result * inputs[i]

        if v is None:
            return result

        # Differential pushforward. For input i with f_i(x_i):
        #   df_i/dx_i = -1/x_i^2  if inv[i] else 1
        # The full product factors as r = f_i(x_i) * P_i where P_i is the
        # product over j != i (including the scaling A), so
        #   dr/dx_i = (df_i/dx_i) * P_i.
        # Mirrors the C++ ``set_value`` ``dout_din`` loop verbatim: for each
        # i, the partial-product coefficient is rebuilt by starting from
        # ``-A/x_i^2`` (reciprocal) or ``A`` (non-reciprocal) and multiplying
        # in every other variable. Done this way (rather than the closed-form
        # ``r / x_i`` shortcut) so that a non-reciprocal ``x_i = 0`` input
        # still yields the correct ``dr/dx_i`` (the C++ regression fixture
        # ``ScalarMultiplication5.i`` exercises this path).
        actions: dict[str, ChainRuleAction] = {}

        def _add(name: str, action: ChainRuleAction) -> None:
            if name in actions:
                prev = actions[name]
                actions[name] = lambda V, a=prev, b=action: a(V) + b(V)
            else:
                actions[name] = action

        for i, fv in enumerate(self._from_vars):
            xi = inputs[i]
            coeff = -A / (xi * xi) if self._inv[i] else A
            for j in range(n):
                if i == j:
                    continue
                xj = inputs[j]
                coeff = coeff / xj if self._inv[j] else coeff * xj
            _add(fv, lambda V, c=coeff: c * V)

        # If scaling was promoted to a nonlinear input, also push its action:
        #   dr/dA = r / A  (linear in A) -> action(V) = (r / A) * V
        nlp_A = self._nl_params.get("scaling")
        if nlp_A is not None:
            coeff_A = result / A
            _add(nlp_A.input_name, lambda V, c=coeff_A: c * V)

        if v2 is None and vh is None:
            return result, *self.propagate_tangents(v, self._to, actions, output=result)

        # ------------------------------------------------------------------
        # Second-order. Each action_2 receives primal-shape
        # Scalar tangents (Va, Vb) and returns a primal-shape Scalar bilinear.
        # The framework's _apply_action_2 iterates the (N_a, N_b) seed-pair
        # outer and stacks; this body is pure typed-wrapper algebra.
        #
        # For the multilinear product r = A · Π_k f_k(x_k):
        #   ∂²r / ∂x_i ∂x_j (i ≠ j) = A · ∂f_i · ∂f_j · Π_{k ∉ {i,j}} f_k.
        #   ∂²r / ∂x_i² = 0 (non-reciprocal) or 2A/x_i³ · Π_{k ≠ i} f_k (reciprocal).
        #   ∂²r / ∂A ∂x_i = ∂f_i · Π_{k ≠ i} f_k.
        #   ∂²r / ∂A² = 0.
        actions_2: dict = {}

        def _add_action_2(a: str, b: str, fn) -> None:
            key = (a, b)
            if key in actions_2:
                prev = actions_2[key]
                actions_2[key] = lambda Va, Vb, p=prev, q=fn: p(Va, Vb) + q(Va, Vb)
            else:
                actions_2[key] = fn

        # (1) Cross-partials (i != j) for non-reciprocal / reciprocal mix.
        for i, fv_i in enumerate(self._from_vars):
            for j, fv_j in enumerate(self._from_vars):
                if i == j:
                    continue
                xi, xj = inputs[i], inputs[j]
                coeff: Scalar = -A / (xi * xi) if self._inv[i] else A  # type: ignore[assignment]
                coeff = -coeff / (xj * xj) if self._inv[j] else coeff
                for k in range(n):
                    if k in (i, j):
                        continue
                    xk = inputs[k]
                    coeff = coeff / xk if self._inv[k] else coeff * xk
                _add_action_2(fv_i, fv_j, lambda Va, Vb, c=coeff: c * Va * Vb)

        # (2) Same-variable second partials (only fire for reciprocal vars).
        for i, fv_i in enumerate(self._from_vars):
            if not self._inv[i]:
                continue
            xi = inputs[i]
            coeff = 2.0 * A / (xi * xi * xi)
            for k in range(n):
                if k == i:
                    continue
                xk = inputs[k]
                coeff = coeff / xk if self._inv[k] else coeff * xk
            _add_action_2(fv_i, fv_i, lambda Va, Vb, c=coeff: c * Va * Vb)

        # (3) Cross-partials with promoted scaling A.
        if nlp_A is not None:
            A_name = nlp_A.input_name
            for i, fv_i in enumerate(self._from_vars):
                xi = inputs[i]
                # ∂f_i / ∂x_i = -1/x_i² (reciprocal) else 1.
                # Use a Union-typed local distinct from the ``coeff: Scalar``
                # above so pyright accepts the ``None`` placeholder branch.
                cross_coeff: Scalar | None = -1.0 / (xi * xi) if self._inv[i] else None
                # When non-reciprocal, the ∂f_i factor is 1, so coeff starts at
                # Π_{k ≠ i} f_k.
                started = cross_coeff is not None
                for k in range(n):
                    if k == i:
                        continue
                    xk = inputs[k]
                    fk = (1.0 / xk) if self._inv[k] else xk
                    cross_coeff = (cross_coeff * fk) if started else fk  # type: ignore[operator]
                    started = True
                if not started:  # n == 1 + non-reciprocal i ⇒ coeff is just 1
                    fn_cross = lambda Va, Vb: Va * Vb  # noqa: E731
                else:
                    fn_cross = lambda Va, Vb, c=cross_coeff: c * Va * Vb  # noqa: E731
                _add_action_2(A_name, fv_i, fn_cross)
                _add_action_2(fv_i, A_name, fn_cross)

        return result, *self.propagate_tangents(
            v, self._to, actions, output=result, v2=v2, actions_2=actions_2, vh=vh
        )


__all__ = ["ScalarMultiplication"]
