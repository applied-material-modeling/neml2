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

"""Forward-guard tests (DECISION.md D-060).

The guard bans AOTI-incompatible / discouraged ``torch`` calls *while a
Python-native ``Model.forward`` is on the stack*: autograd & functional
transforms (hard) and ``torch.einsum`` (hard), each liftable via an
``allow_*`` context manager that requires a written reason. Outside a forward
(``depth == 0``) the patched callables are transparent.
"""

from __future__ import annotations

import pytest
import torch

from neml2 import allow_autograd, allow_einsum
from neml2.models import _guard as guard
from neml2.models.common import ComposedModel
from neml2.models.model import Model
from neml2.types import Scalar


# --------------------------------------------------------------------------- #
# Minimal leaves
# --------------------------------------------------------------------------- #
class _Clean(Model):
    """``b = a**2`` — no banned calls."""

    input_spec = {"a": Scalar}
    output_spec = {"b": Scalar}

    def forward(self, *inputs, v=None):  # type: ignore[override]
        (a,) = inputs
        return Scalar(a.data**2)


def _leaf_calling(fn):
    """Build a leaf whose forward invokes ``fn(a.data)`` (an offending call)."""

    class _Bad(Model):
        input_spec = {"a": Scalar}
        output_spec = {"b": Scalar}

        def forward(self, *inputs, v=None):  # type: ignore[override]
            (a,) = inputs
            fn(a.data)
            return Scalar(a.data)

    return _Bad()


_ONE = Scalar(torch.ones(()))


# --------------------------------------------------------------------------- #
# Forbidden calls inside forward raise
# --------------------------------------------------------------------------- #
def test_einsum_in_forward_raises():
    leaf = _leaf_calling(lambda t: torch.einsum("...,...->...", t, t))
    with pytest.raises(RuntimeError, match="torch.einsum is disallowed"):
        leaf(_ONE)


def test_autograd_grad_in_forward_raises():
    leaf = _leaf_calling(lambda t: torch.autograd.grad(t.sum(), t))
    with pytest.raises(RuntimeError, match="not compatible with torch.export"):
        leaf(_ONE)


def test_tensor_backward_in_forward_raises():
    leaf = _leaf_calling(lambda t: t.sum().backward())
    with pytest.raises(RuntimeError, match="not compatible with torch.export"):
        leaf(_ONE)


def test_func_jacfwd_in_forward_raises():
    # Constructing the transform is enough — the wrapper checks on call.
    leaf = _leaf_calling(lambda t: torch.func.jacfwd(lambda z: z * z))
    with pytest.raises(RuntimeError, match="torch.func.jacfwd"):
        leaf(_ONE)


def test_enable_grad_in_forward_raises():
    leaf = _leaf_calling(lambda t: torch.enable_grad())
    with pytest.raises(RuntimeError, match="torch.enable_grad"):
        leaf(_ONE)


# --------------------------------------------------------------------------- #
# Escape hatches
# --------------------------------------------------------------------------- #
def test_allow_einsum_permits_inside_forward():
    leaf = _leaf_calling(lambda t: torch.einsum("...,...->...", t, t))
    with allow_einsum("unit test: deliberately exercising the escape hatch"):
        leaf(_ONE)  # no raise


def test_allow_autograd_permits_inside_forward():
    class _Adjoint(Model):
        input_spec = {"a": Scalar}
        output_spec = {"b": Scalar}

        def forward(self, *inputs, v=None):  # type: ignore[override]
            (a,) = inputs
            t = a.data.detach().requires_grad_(True)
            with allow_autograd("unit test: eager IFT-style adjoint"):
                (g,) = torch.autograd.grad((t * t).sum(), t)
            return Scalar(g.sum())

    out = _Adjoint()(Scalar(torch.tensor(3.0)))
    assert torch.allclose(out.data, torch.tensor(6.0))  # d(t^2)/dt = 2t = 6


@pytest.mark.parametrize("allow", [allow_autograd, allow_einsum])
@pytest.mark.parametrize("reason", ["", "   ", None])
def test_allow_requires_nonblank_reason(allow, reason):
    with pytest.raises(ValueError, match="non-empty reason"):
        with allow(reason):  # type: ignore[arg-type]
            pass


# --------------------------------------------------------------------------- #
# No false positives outside a forward (depth 0)
# --------------------------------------------------------------------------- #
def test_no_false_positive_at_depth_zero():
    # einsum and autograd.grad work normally when no forward is on the stack.
    assert torch.einsum("i,i->", torch.ones(3), torch.ones(3)).item() == 3.0
    x = torch.randn(3, requires_grad=True)
    (g,) = torch.autograd.grad((x * x).sum(), x)
    assert torch.allclose(g, 2 * x)


def test_autograd_functional_jacobian_reference_still_works():
    # A common test idiom: differentiate a *clean* model with autograd as an
    # independent reference. The call sits at depth 0; the model forward opens
    # and closes its own window, so the surrounding autograd never trips.
    m = _Clean()
    J = torch.autograd.functional.jacobian(lambda t: m(Scalar(t)).data, torch.tensor(2.0))
    assert torch.allclose(J, torch.tensor(4.0))  # d(t^2)/dt = 2t = 4


# --------------------------------------------------------------------------- #
# Nesting + bookkeeping
# --------------------------------------------------------------------------- #
def test_nested_composed_model_still_guarded():
    bad = _leaf_calling(lambda t: torch.einsum("...,...->...", t, t))
    cm = ComposedModel([bad])
    with pytest.raises(RuntimeError, match="torch.einsum is disallowed"):
        cm(torch.ones(()))


def test_clean_forward_runs_under_guard():
    out = _Clean()(Scalar(torch.tensor(3.0)))
    assert torch.allclose(out.data, torch.tensor(9.0))


def test_depth_balanced_after_exception():
    leaf = _leaf_calling(lambda t: torch.einsum("...,...->...", t, t))
    with pytest.raises(RuntimeError):
        leaf(_ONE)
    assert guard._depth() == 0
