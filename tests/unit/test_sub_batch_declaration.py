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
from neml2.settings import model_variable_names
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


def test_rewrapping_accepts_an_explicit_value_that_agrees():
    inner = Scalar(torch.zeros(2, 3, dtype=torch.float64), sub_batch_ndim=1)
    assert Scalar(inner, 1).sub_batch_ndim == 1
    # ``()`` is the documented shorthand for all-"full", so spelling it out is
    # the same statement, not a competing one.
    assert Scalar(inner, 1, ("full",)).sub_batch_state == ("full",)


@pytest.mark.parametrize("ndim", [0, 2], ids=["drop", "bump"])
def test_rewrapping_refuses_an_explicit_value_that_disagrees(ndim):
    """A re-wrap that is handed a rank *and* a value carrying a different one
    has two answers and no way to choose. Preferring the argument drops the
    region; preferring the value ignores the argument; taking the argument's
    rank with the value's per-axis state builds an object neither described
    (a rank-2 wrapper with a one-element ``sub_batch_state``, which crashed
    ``sub_batch_shape``). So it refuses.

    ``ndim=0`` is the case that has to raise rather than be treated as
    silence: 0 is also the no-region default, and reading an explicit 0 as
    "said nothing" is what let a per-site quantity keep its axis through a
    re-wrap that meant to strip it.
    """
    inner = Scalar(torch.zeros(2, 3, dtype=torch.float64), sub_batch_ndim=1)
    with pytest.raises(ValueError, match=r"Cannot re-wrap a Scalar.*sub_batch_ndim"):
        Scalar(inner, ndim)


def test_dropping_a_region_has_its_own_spelling():
    """The escape hatch the refusal above points at: unambiguous by
    construction, because there is no second opinion in the call."""
    inner = Scalar(torch.zeros(2, 3, dtype=torch.float64), sub_batch_ndim=1)
    assert inner.with_sub_batch_ndim(0).sub_batch_ndim == 0
    assert Scalar(inner.data).sub_batch_ndim == 0


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

#: Declares the predictor-owned unknown by its bare name -- the spelling the
#: docs teach and ``per_slip_hardening_declared`` uses.
_DECLARE_U_PER_SITE = f"""
[Settings]
  [example_batch_shape]
    u = '(2; {_NSITE})'
  []
[]
"""


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


# ---------------------------------------------------------------------------
# TransientDriver: the declaration sizes what the driver has to invent
# ---------------------------------------------------------------------------

# A per-site unknown integrated in time, driven with a global rate. The initial
# condition is a plain value -- NOT hand-shaped to (nbatch, nsite) -- so the
# only thing that knows the per-site extent is the [Settings] declaration.
_DRIVEN = """
[Tensors]
  [times]
    type = Python
    expr = '''
      t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
      result = Scalar(t.unsqueeze(-1).expand(5, 3).contiguous())
    '''
  []
  [rates]
    type = Python
    expr = 'Scalar(torch.full((5, 3), 2.0, dtype=torch.float64))'
  []
  [x0]
    type = Python
    expr = 'Scalar(10.0)'
  []
  [x0_per_site]
    type = Python
    expr = 'Scalar(torch.full((3, 4), 10.0, dtype=torch.float64), sub_batch_ndim=1)'
  []
  [x0_wrong_width]
    type = Python
    expr = 'Scalar(torch.full((3, 7), 10.0, dtype=torch.float64), sub_batch_ndim=1)'
  []
  # Unmarked, with batch axes: ambiguous against a declared sub-batch extent.
  # The second one is the trap -- its trailing axis is the declared 4.
  [x0_batched]
    type = Python
    expr = 'Scalar(torch.full((3,), 10.0, dtype=torch.float64))'
  []
  [x0_batched_coincidental]
    type = Python
    expr = 'Scalar(torch.full((3, 4), 10.0, dtype=torch.float64))'
  []
[]

[Models]
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'integrate'
    unknowns = 'x'
    residuals = 'x_residual'
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
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_Scalar_names = 'x_rate'
    prescribed_Scalar_values = 'rates'
    ic_Scalar_names = 'x'
    ic_Scalar_values = '{ic}'
  []
[]
{settings}
"""

_DECLARE_PER_SITE = """
[Settings]
  [example_batch_shape]
    x = '(2; 4)'
  []
[]
"""


def _driver(ic: str = "x0", settings: str = _DECLARE_PER_SITE):
    return load_string(_DRIVEN.format(ic=ic, settings=settings)).get_driver("driver")


def test_driver_reads_the_declaration_for_the_variable_and_its_lags():
    driver = _driver()
    assert driver.declared_sub_batch_shapes == {
        "x": torch.Size([4]),
        "x~1": torch.Size([4]),
    }


def test_driver_sizes_its_zero_fill_from_the_declaration():
    """``x`` is an unknown with no predictor, so the driver has to invent its
    initial guess. Zero-filled at base shape it gives the equation system one
    row where the residual has one per site, and the linear solve dies on a
    non-square matrix -- the reason `predictor = ` omitted, a schema-optional
    configuration, was not usable for a sub-batched system."""
    driver = _driver()
    assert driver.run()
    final = driver.result_out[-1]["x"]
    assert final.sub_batch_ndim == 1
    assert tuple(final.shape) == (3, 4)
    # x0 = 10 broadcast over the 4 sites, integrated at rate 2 to t = 1.
    torch.testing.assert_close(final.data, torch.full((3, 4), 12.0, dtype=torch.float64))


def test_declared_and_hand_shaped_ic_agree_on_the_answer():
    """The behaviour the declaration exists to replace: pinning the shape with
    a hand-built ``(nbatch, nsite)`` initial condition. Both spellings solve,
    and they solve to the same thing."""
    declared = _driver()
    declared.run()
    hand_shaped = _driver(ic="x0_per_site", settings="")
    hand_shaped.run()
    torch.testing.assert_close(
        declared.result_out[-1]["x"].data, hand_shaped.result_out[-1]["x"].data
    )


def test_driver_rejects_an_ic_that_contradicts_the_declaration():
    with pytest.raises(ValueError, match=r"declares x sub_batch=\(4,\).*carries sub_batch=\(7,\)"):
        _driver(ic="x0_wrong_width")


@pytest.mark.parametrize("ic", ["x0_batched", "x0_batched_coincidental"], ids=["(3,)", "(3,4)"])
def test_driver_rejects_an_unmarked_ic_that_has_batch_axes(ic):
    """An unmarked ``(3, 4)`` could be three batch members of four sites or a
    twelve-element batch; nothing in the value says which. Reading it as batch
    appends the declared axes and silently yields ``(3, 4, 4)``, so both
    spellings are refused rather than guessed at. ``x0_batched_coincidental``
    is the dangerous one -- its trailing axis matches the declaration, so a
    guess would look right."""
    with pytest.raises(ValueError, match=r"declares x sub_batch=\(4,\).*unmarked but has batch"):
        _driver(ic=ic)


def test_driver_accepts_an_ic_that_already_matches_the_declaration():
    driver = _driver(ic="x0_per_site")
    assert driver.declared_sub_batch_shapes["x"] == torch.Size([4])
    assert driver.run()


def test_undeclared_sub_batched_variable_still_infers_from_its_ic():
    """No declaration: the pre-existing inference from a hand-shaped initial
    condition keeps working, so files that predate this keep their behaviour."""
    driver = _driver(ic="x0_per_site", settings="")
    assert driver.declared_sub_batch_shapes == {}
    assert driver.run()
    assert tuple(driver.result_out[-1]["x"].shape) == (3, 4)


# ---------------------------------------------------------------------------
# Route parity
# ---------------------------------------------------------------------------


def test_eager_and_export_resolve_the_same_sub_batch_extents():
    """Parity is an invariant (CLAUDE.md): a declaration that means one thing
    to the driver and another to ``neml2-compile`` is worse than no
    declaration, because the two routes would then disagree silently. Both
    resolve through :class:`~neml2.settings.Settings`; this pins that neither
    has grown a private interpretation of the same file.
    """
    factory = load_string(_DRIVEN.format(ic="x0", settings=_DECLARE_PER_SITE))
    model = factory.get_model("model")
    driver = factory.get_driver("driver")
    system = model.system

    # What ``neml2-compile`` would trace, i.e. the exporter's resolution.
    exported = factory.settings.resolve(model.input_spec, model_variable_names(model))

    for name in model.input_spec:
        declared = driver.declared_sub_batch_shapes.get(name)
        if declared is None:
            continue
        assert tuple(exported[name][1]) == tuple(declared), name
        assert tuple(system.declared_sub_batch_shapes[name]) == tuple(declared), name


def test_export_accepts_a_declaration_for_a_predictor_owned_unknown():
    """The case the parity test above cannot reach. ``_DRIVEN`` has no
    predictor, so its unknown ``x`` is also an *input* of the ImplicitUpdate
    and any universe that includes ``input_spec`` covers it. Give the unknown a
    predictor and it becomes an output owned by nobody's input_spec -- yet
    declaring its extent is the whole reason the declaration exists, and
    ``dislocation_density`` in ``per_slip_hardening_declared`` is spelled
    exactly this way. Validating against ``input_spec`` alone made the natural
    spelling of the natural case a hard error on the compiled route while the
    eager route accepted it.
    """
    factory = load_string(_PER_SITE_WITH_PREDICTOR + _DECLARE_U_PER_SITE)
    model = factory.get_model("model")
    assert "u" not in model.input_spec, "u must be predictor-owned for this test to bite"

    # Resolution must not reject the declaration, and the lag family must carry
    # the extent through to what actually gets traced.
    resolved = factory.settings.resolve(model.input_spec, model_variable_names(model))
    assert factory.settings.sub_batch_shape("u") == (3,)
    assert all(sub == (3,) for _, sub in resolved.values() if sub)

    # And the eager side agrees on the same file.
    assert tuple(model.system.declared_sub_batch_shapes["u"]) == (3,)


def test_a_declaration_for_a_name_no_variable_has_is_rejected_on_both_routes():
    """A mistyped declaration describes nothing, so it is silent by
    construction -- the variable it was meant to size keeps its inferred shape
    and the run either works by luck or fails somewhere unrelated. Both routes
    check against the same universe so neither can be the lenient one."""
    typo = _DECLARE_U_PER_SITE.replace("u =", "you =")
    factory = load_string(_PER_SITE_WITH_PREDICTOR + typo)
    model = factory.get_model("model")
    with pytest.raises(ValueError, match=r"names not in .*: \['you'\]"):
        factory.settings.resolve(model.input_spec, model_variable_names(model))

    with pytest.raises(ValueError, match=r"names not in .*: \['nope'\]"):
        _driver(settings=_DECLARE_PER_SITE.replace("x =", "nope ="))
