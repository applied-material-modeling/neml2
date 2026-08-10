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

"""How a sub-batched variable's shape becomes known before a tensor for it exists.

A sub-batch axis (per slip system, per bin, per site) is a static property of a
variable, but nothing in the eager path used to *declare* it -- the rank was
inferred from whichever tensor happened to arrive first, and silently defaulted
to zero when none did. These tests pin the two things that replaced that guess:
the signals :func:`~neml2.models.common.ImplicitUpdate._resolve_unknown_sbn`
combines, and the ``[Settings]/example_batch_shape`` declaration the eager
consumers read.
"""

from __future__ import annotations

import pytest
import torch

import neml2  # noqa: F401 -- registers the model types the HIT snippets name
from neml2.factory import load_string
from neml2.models.common.ImplicitUpdate import _resolve_unknown_sbn
from neml2.types import SR2, Scalar

# ---------------------------------------------------------------------------
# Re-wrapping a typed value keeps its metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "shape"),
    [(Scalar, (2, 3)), (SR2, (2, 3, 6))],
    ids=["Scalar", "SR2"],
)
def test_rewrapping_a_wrapper_inherits_its_sub_batch_ndim(cls, shape):
    """``type_cls(wrapper)`` is how a value crosses a model boundary. It used
    to reset the sub-batch region to the constructor default, so every caller
    that cared had to re-attach the rank by hand -- and the ones that forgot
    (``Model.call_by_name``, and so every predictor) handed a per-site quantity
    downstream as a plain batched one."""
    inner = cls(torch.zeros(*shape, dtype=torch.float64), sub_batch_ndim=1)
    assert cls(inner).sub_batch_ndim == 1


def test_rewrapping_still_lets_an_explicit_value_win():
    inner = Scalar(torch.zeros(2, 3, dtype=torch.float64), sub_batch_ndim=1)
    assert Scalar(inner, 2).sub_batch_ndim == 2
    # Dropping the region is available, but has to be asked for.
    assert inner.with_sub_batch_ndim(0).sub_batch_ndim == 0


def test_wrapping_a_raw_tensor_still_defaults_to_zero():
    assert Scalar(torch.zeros(2, 3, dtype=torch.float64)).sub_batch_ndim == 0


def test_call_by_name_does_not_strip_sub_batch_from_its_inputs():
    """``call_by_name`` rewraps every input through the input_spec; that is the
    hop where a predictor's per-site input used to arrive unmarked."""
    model = load_string(_PER_SITE_WITH_PREDICTOR).get_model("warm_start")
    out = model.call_by_name({"g": _per_site_given()})
    assert out["u"].sub_batch_ndim == 1


# ---------------------------------------------------------------------------
# _resolve_unknown_sbn: combining the available signals
# ---------------------------------------------------------------------------


def test_resolve_unknown_sbn_defaults_to_zero_when_nothing_asserts():
    assert _resolve_unknown_sbn("x", produced=Scalar(torch.zeros(())), input_sbn={}) == 0


def test_resolve_unknown_sbn_reads_the_history_input():
    """The pre-existing signal: ``x~1`` is the same variable one step back."""
    produced = Scalar(torch.zeros(2, 4))
    assert _resolve_unknown_sbn("x", produced=produced, input_sbn={"x~1": 1}) == 1
    assert _resolve_unknown_sbn("x", produced=produced, input_sbn={"x~3": 1}) == 1


def test_resolve_unknown_sbn_reads_the_initial_guess():
    """The signal that used to be discarded: a predictor produces the unknown,
    so the unknown is not an input of ImplicitUpdate at all and its own
    metadata is the only thing that knows the rank."""
    produced = Scalar(torch.zeros(2, 12), sub_batch_ndim=1)
    assert _resolve_unknown_sbn("slip_rates", produced=produced, input_sbn={}) == 1


def test_resolve_unknown_sbn_ignores_unrelated_variables():
    produced = Scalar(torch.zeros(2))
    assert _resolve_unknown_sbn("x", produced=produced, input_sbn={"y": 1, "y~1": 1}) == 0


def test_resolve_unknown_sbn_treats_zero_as_silence_not_contradiction():
    """A rank of 0 is indistinguishable from unmarked, so it can never override
    a positive assertion -- otherwise a base-shaped seed would erase the axis."""
    produced = Scalar(torch.zeros(2, 4))
    assert _resolve_unknown_sbn("x", produced=produced, input_sbn={"x": 0, "x~1": 1}) == 1


def test_resolve_unknown_sbn_rejects_conflicting_positive_ranks():
    produced = Scalar(torch.zeros(2, 3, 4), sub_batch_ndim=2)
    with pytest.raises(ValueError, match="conflicting sub-batch ranks for unknown 'x'"):
        _resolve_unknown_sbn("x", produced=produced, input_sbn={"x~1": 1})


# ---------------------------------------------------------------------------
# End-to-end: a predictor that produces a sub-batched unknown
# ---------------------------------------------------------------------------

# One per-site unknown with NO time history, so its rank cannot be recovered
# from a ``u~1`` input -- the crystal-plasticity ``slip_rates`` shape, reduced
# to the smallest system that reproduces it. Closed form: u = g.
_PER_SITE_WITH_PREDICTOR = """
[Models]
  [r]
    type = ScalarLinearCombination
    from = 'u g'
    to = 'r'
    weights = '1 -1'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'r'
    unknowns  = 'u'
    residuals = 'r'
  []
[]

[Solvers]
  [lu]
    type = DenseLU
  []
  [newton]
    type = Newton
    linear_solver = 'lu'
    abs_tol = 1e-12
    rel_tol = 1e-10
    max_its = 25
  []
[]

[Models]
  # A warm start for the per-site unknown, chained off the per-site given.
  # Its output carries sub_batch_ndim=1 -- and because the predictor produces
  # `u`, `u` is no longer an input of the ImplicitUpdate, so this wrapper is
  # the only thing left that knows the rank.
  [warm_start]
    type = ScalarLinearCombination
    from = 'g'
    to = 'u'
    weights = '0.5'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'warm_start'
  []
[]
"""

_NBATCH = 2
_NSITE = 3


def _per_site_given() -> Scalar:
    return Scalar(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).expand(_NBATCH, _NSITE).contiguous(),
        sub_batch_ndim=1,
    )


def test_predictor_output_keeps_its_sub_batch_axis():
    """Regression: the predictor's ``u`` reached the equation system with its
    sub-batch axis stripped, because the unknown's rank was looked up only
    through history inputs and this unknown has none. The unknown vector then
    had one row where the residual had one per site."""
    model = load_string(_PER_SITE_WITH_PREDICTOR).get_model("model")
    assert "u" not in model.input_spec, (
        "the predictor is supposed to own u; if it is still an input the test "
        "is no longer exercising the discarded-metadata path"
    )

    out = model.call_by_name({"g": _per_site_given()})

    assert out["u"].sub_batch_ndim == 1
    assert tuple(out["u"].shape) == (_NBATCH, _NSITE)
    torch.testing.assert_close(out["u"].data, _per_site_given().data)
