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

"""Python-native mirror of C++ ``common/ScalarPNorm.h``."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, output, parameter, parameters, var_inputs
from ...types import Scalar, abs, pow, sign


@register_native("ScalarPNorm")
class ScalarPNorm(Model):
    r"""Weighted $p$-norm of an arbitrary number of Scalar inputs:
    $y = (\sum_i w_i |x_i|^p + \varepsilon)^{1/p}$. The weights default
    to 1 (giving the standard $p$-norm) and can be set per-input via the
    `weights` option, mirroring `LinearCombination`. The dtype-aware
    regularizer $\varepsilon$ comes from `neml2::machine_precision()`.
    """

    # The exponent ``p`` is forward-only on the C++ side (allow_nonlinear=false)
    # because the C++ set_value() never assigns _to.d(*p); we keep the same
    # static-only contract here. The weights mirror the C++ buffer-or-parameter
    # toggle implicitly: every coefficient is a parameter on the native side
    # (same simplification as LinearCombination).
    hit = HitSchema(
        var_inputs("from", Scalar, "Scalar variables to be combined", attr="_from_vars"),
        output("to", Scalar, "The weighted p-norm output", attr="_to"),
        parameter("exponent", Scalar, "The exponent", attr="_p"),
        parameters(
            "weights",
            Scalar,
            "Per-input weights. List length must be 1 or `from`-length; a single value is "
            "broadcast to all inputs.",
            default=["1"],
            attr="weight",
        ),
    )

    _from_vars: list[str]
    _to: str
    weight: list[str]
    _p: Scalar

    def __post_init__(self) -> None:
        # Mirror the C++ ``weights`` length check: 1 (broadcast) or
        # ``len(from)`` one-for-one. Run after the ``parameters`` schema field
        # populates ``self.weight`` (pass 1) but before the deferred
        # parameter declarations (pass 2) — same staging as LinearCombination.
        n = len(self._from_vars)
        if len(self.weight) != 1 and len(self.weight) != n:
            raise ValueError(
                f"{type(self).__name__}: weights must have length 1 or {n}, got "
                f"{len(self.weight)}: {self.weight!r}"
            )

    def forward(  # type: ignore[override]
        self,
        *args: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Split positional inputs: the leading structural inputs (one per
        # from-var) followed by the *nl_params pack that holds any
        # mode-3/4-promoted parameters. Only weights may promote; the
        # exponent stays static (allow_nonlinear=False).
        n_from = len(self._from_vars)
        inputs, nl_params = args[:n_from], args[n_from:]
        weights = self._get_param_list("weight", nl_params, Scalar)
        if len(weights) == 1:
            weights = weights * n_from
        p = self._p

        # Sum_i w_i * |x_i|^p. With ``p`` held static (allow_nonlinear=False),
        # autograd never differentiates ``pow`` w.r.t. the exponent, so the
        # ``where``-mask-and-safe-base trick the C++ side needs to guard
        # ``d/dp [x^p] = x^p log(x)`` against the log(0) singularity is
        # unnecessary here. The forward-only autograd path through
        # ``pow(|x|, p)`` is safe at x=0 for p >= 1 (gives 0, matching the
        # analytical limit).
        eps = torch.finfo(inputs[0].dtype).eps
        terms = [w * pow(abs(x), p) for w, x in zip(weights, inputs, strict=True)]
        s = terms[0]
        for t in terms[1:]:
            s = s + t

        # y = (sum + eps)^(1/p). The regularizer keeps y bounded below by
        # eps^(1/p) > 0, so 1/y in the Jacobian stays finite.
        inv_p = 1.0 / p
        y = pow(s + eps, inv_p)

        if v is None:
            return y

        # D-062: per-input pushforward of y(x_1, ..., x_n).
        #   dy/dx_i = w_i * sign(x_i) * |x_i|^(p-1) * y^(1-p)
        # so action_i(V) = (w_i * sign(x_i) * |x_i|^(p-1) * y^(1-p)) * V.
        # For p=2 with unit weights this collapses to x_i / y * V.
        # Precompute the y^(1-p) factor (shared across all inputs).
        y_pow = pow(y, 1.0 - p)
        p_minus_one = p - 1.0

        def _make_action(coef: Scalar) -> Callable[[Scalar], Scalar]:
            def action(V: Scalar) -> Scalar:
                return coef * V

            return action

        actions: dict[str, Callable[[Scalar], Scalar]] = {}
        for fv, w, xi in zip(self._from_vars, weights, inputs, strict=True):
            xi_factor = w * sign(xi) * pow(abs(xi), p_minus_one) * y_pow
            actions[fv] = _make_action(xi_factor)

        return y, self.apply_chain_rule(v, self._to, actions, output=y)


__all__ = ["ScalarPNorm"]
