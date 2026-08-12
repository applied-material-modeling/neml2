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

"""Unit tests for :mod:`neml2.settings` -- the route-neutral ``[Settings]`` reader."""

from __future__ import annotations

import pytest

from neml2.factory import load_string
from neml2.settings import (
    DEFAULT_EXAMPLE_SHAPE,
    Settings,
    parse_example_batch_shape,
    resolve_example_shapes,
    sub_batch_conflict,
    validate_declared_names,
)

# ---------------------------------------------------------------------------
# Spec grammar
# ---------------------------------------------------------------------------


def test_parse_example_batch_shape_no_sub_batch():
    # No sub-batch axes: dyn-only specs return an empty sub tuple.
    assert parse_example_batch_shape("(2,)") == ((2,), ())
    assert parse_example_batch_shape("(2, 3)") == ((2, 3), ())


def test_parse_example_batch_shape_with_sub_batch():
    # Semicolon splits dyn from sub.
    assert parse_example_batch_shape("(2; 3)") == ((2,), (3,))
    assert parse_example_batch_shape("(2; 3, 12)") == ((2,), (3, 12))
    # Empty dyn region is allowed (static-batch + sub).
    assert parse_example_batch_shape("(; 100)") == ((), (100,))


def test_parse_example_batch_shape_rejects_label_suffix():
    """The ``:label`` suffix on sub-batch extents was removed in V2P-9
    (the chain rule no longer dispatches on labels). A leftover ``:foo``
    must be flagged with a clear error rather than silently parsed."""
    with pytest.raises(ValueError, match=":label.*removed"):
        parse_example_batch_shape("(2; 3:grain)")


def test_parse_example_batch_shape_requires_parens():
    with pytest.raises(ValueError, match="must be parenthesized"):
        parse_example_batch_shape("2; 3")


# ---------------------------------------------------------------------------
# Eager-side lookup: sub_batch_shape / sub_batch_ndim
# ---------------------------------------------------------------------------


def test_sub_batch_shape_is_none_when_undeclared():
    """``None`` (nothing said) and ``()`` (said: no sub-batch) are different
    answers -- a consumer falls back to inference only on ``None``."""
    assert Settings().sub_batch_shape("slip_rates") is None
    assert Settings().sub_batch_ndim("slip_rates") is None
    assert Settings({"other": "(2; 4)"}).sub_batch_shape("slip_rates") is None


def test_sub_batch_shape_exact_key_wins():
    s = Settings({"slip_rates": "(2; 12)", "*": "(8,)"})
    assert s.sub_batch_shape("slip_rates") == (12,)
    assert s.sub_batch_ndim("slip_rates") == 1


def test_sub_batch_shape_falls_back_to_the_lag_family():
    """A declaration on any lag of a variable governs every lag of it: the
    sub-batch axis belongs to the variable, not to the time step."""
    s = Settings({"elastic_strain~1": "(2; 20)"})
    assert s.sub_batch_shape("elastic_strain~1") == (20,)
    assert s.sub_batch_shape("elastic_strain") == (20,)
    assert s.sub_batch_shape("elastic_strain~2") == (20,)


def test_a_spec_without_a_semicolon_declares_no_sub_batch():
    """``'(N,)'`` writes a dynamic region only. Reading it as "and therefore no
    sub-batch axis anywhere" would break every benchmark input, which uses the
    uniform form purely to pin a production batch size."""
    assert Settings({"*": "(8,)"}).sub_batch_shape("anything") is None
    assert Settings({"x": "(8,)"}).sub_batch_shape("x") is None


def test_an_explicitly_empty_sub_region_is_a_claim():
    """``'(2; )'`` writes the region and leaves it empty -- that *is* an
    assertion that the variable has no sub-batch axis."""
    assert Settings({"x": "(2; )"}).sub_batch_shape("x") == ()
    assert Settings({"x": "(2; )"}).sub_batch_ndim("x") == 0


def test_uniform_entry_can_declare_a_sub_batch():
    assert Settings({"*": "(8; 5)"}).sub_batch_shape("anything") == (5,)


def test_lag_entries_must_agree_on_the_sub_region():
    with pytest.raises(ValueError, match="history lags must declare the same sub-batch"):
        Settings({"x": "(2; 4)", "x~1": "(2; 8)"})


def test_a_silent_lag_does_not_conflict_with_a_declaring_one():
    """``benchmark/mxpc/model.i`` shape: some entries pin a sub-batch, others
    only pin a dynamic batch. The silent ones borrow, not fight."""
    s = Settings({"x": "(2,)", "x~1": "(2; 12)"})
    assert s.sub_batch_shape("x") == (12,)
    assert s.sub_batch_shape("x~1") == (12,)


def test_lag_entries_may_differ_on_the_dyn_region():
    """Only the sub region is a property of the variable; the dynamic region
    is a per-input trace hint and may legitimately differ across lags (see
    ``benchmark/mxpc/model.i``)."""
    s = Settings({"x": "(4; 12)", "x~1": "(2; 12)"})
    assert s.sub_batch_shape("x") == (12,)


# ---------------------------------------------------------------------------
# Export-side resolution
# ---------------------------------------------------------------------------


def test_resolve_assigns_the_default_when_nothing_is_declared():
    assert resolve_example_shapes(["a", "b"], {}) == {
        "a": DEFAULT_EXAMPLE_SHAPE,
        "b": DEFAULT_EXAMPLE_SHAPE,
    }


def test_resolve_prefers_per_variable_over_uniform():
    resolved = resolve_example_shapes(["a", "b"], {"a": "(2; 5)", "*": "(2,)"})
    assert resolved == {"a": ((2,), (5,)), "b": ((2,), ())}


def test_resolve_rejects_names_outside_the_universe():
    with pytest.raises(ValueError, match="names not in model variables: \\['stress'\\]"):
        resolve_example_shapes(["strain"], {"stress": "(2,)"})


def test_validate_declared_names_reports_the_named_universe():
    with pytest.raises(ValueError, match="names not in driver variables"):
        validate_declared_names({"nope": "(2,)"}, ["a"], universe="driver variables")


def test_validate_declared_names_ignores_the_uniform_key():
    validate_declared_names({"*": "(2,)"}, ["a"])


def test_validate_declared_names_accepts_a_lag_of_a_known_variable():
    """A declaration on ``x~1`` governs ``x`` and vice versa, so a spelling
    that resolves has to be a spelling that validates."""
    validate_declared_names({"x~1": "(2; 4)"}, ["x"])
    validate_declared_names({"x": "(2; 4)"}, ["x~1"])


def test_resolve_applies_lag_agreement_to_a_cli_override_map():
    """``--example-batch-shape x='(2;4)' --example-batch-shape 'x~1=(2;8)'``
    reaches :func:`resolve_example_shapes` as a plain dict, bypassing
    :func:`read_settings`. It is the same declaration either way, so it gets
    the same consistency check -- an override that contradicts itself must not
    be quieter than a ``[Settings]`` block that does."""
    with pytest.raises(ValueError, match="history lags must declare the same sub-batch"):
        resolve_example_shapes(["x", "x~1"], {"x": "(2; 4)", "x~1": "(2; 8)"})


def test_resolve_accepts_a_declared_name_outside_input_spec_when_known_is_given():
    """The exporter validates against every variable the model tree mentions,
    because a predictor-owned implicit unknown is an output and declaring its
    extent is exactly what the declaration is for. Resolution still runs over
    ``input_spec``; the unknown reaches the tracer through its ``~1`` lag."""
    resolved = resolve_example_shapes(["u~1"], {"u": "(2; 12)"}, known={"u", "u~1"})
    assert resolved == {"u~1": ((2,), (12,))}


# ---------------------------------------------------------------------------
# Declaration-vs-value compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sub", "batch", "ok"),
    [
        ((4,), (3,), True),  # already carries the declared extent
        ((), (), True),  # a bare seed: the declared axes get added
        ((7,), (3,), False),  # a different extent is a flat contradiction
        ((), (3,), False),  # unmarked with batch axes: ambiguous
        ((), (3, 4), False),  # ... including when the trailing axis matches
    ],
    ids=["matches", "bare-seed", "wrong-extent", "unmarked-batched", "unmarked-coincidental"],
)
def test_sub_batch_conflict(sub, batch, ok):
    """An unmarked value is only expanded when it has no batch axes at all.
    ``unmarked-coincidental`` is why: nothing in a ``(3, 4)`` tensor says
    whether the 4 is four sites or four batch members, and reading it as batch
    appends the declared axes to produce ``(3, 4, 4)``. Guessing right most of
    the time is not good enough for a shape the whole solve is built on.

    :func:`~neml2.models.common.ImplicitUpdate._conform_to_declared_sub_batch`
    applies the same rule -- see
    ``test_sub_batch_declaration.py`` for the paired end-to-end cases.
    """
    reason = sub_batch_conflict((4,), sub_batch_shape=sub, batch_shape=batch)
    assert (reason is None) is ok, reason


# ---------------------------------------------------------------------------
# Reading the block off a HIT file
# ---------------------------------------------------------------------------

_MINIMAL_MODEL = """
[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
    stress = 'cauchy_stress'
  []
[]
"""


def test_read_settings_defaults_without_a_settings_block():
    settings = load_string(_MINIMAL_MODEL).settings
    assert settings.example_batch_shape == {}
    assert settings.dynamic_batch is True
    assert settings.sub_batch_shape("elastic_strain") is None


def test_read_settings_uniform_field_form():
    settings = load_string(
        _MINIMAL_MODEL
        + """
[Settings]
  example_batch_shape = '(4; 12)'
  dynamic_batch = false
[]
"""
    ).settings
    assert settings.dynamic_batch is False
    assert settings.sub_batch_shape("elastic_strain") == (12,)


def test_read_settings_per_variable_section_form():
    settings = load_string(
        _MINIMAL_MODEL
        + """
[Settings]
  [example_batch_shape]
    elastic_strain = '(2; 12)'
    cauchy_stress = '(2,)'
  []
[]
"""
    ).settings
    assert settings.sub_batch_shape("elastic_strain") == (12,)
    # ``'(2,)'`` pins only the dynamic region -- silence about sub-batch.
    assert settings.sub_batch_shape("cauchy_stress") is None


def test_read_settings_rejects_both_forms_at_once():
    with pytest.raises(ValueError, match="cannot use both the field"):
        _ = load_string(
            _MINIMAL_MODEL
            + """
[Settings]
  example_batch_shape = '(2,)'
  [example_batch_shape]
    elastic_strain = '(2; 12)'
  []
[]
"""
        ).settings


def test_read_settings_rejects_a_non_boolean_dynamic_batch():
    with pytest.raises(ValueError, match="expected boolean"):
        _ = load_string(
            _MINIMAL_MODEL
            + """
[Settings]
  dynamic_batch = maybe
[]
"""
        ).settings


def test_factory_caches_the_parsed_settings():
    factory = load_string(_MINIMAL_MODEL)
    assert factory.settings is factory.settings
