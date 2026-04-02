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

import textwrap
from pathlib import Path

import pytest

from neml2.reader._parser import ModelEntry, parse_input

# Path to the Chaboche regression model used as the primary fixture
CHABOCHE = (
    Path(__file__).parents[3] / "tests/regression/solid_mechanics/viscoplasticity/chaboche/model.i"
)


@pytest.fixture
def chaboche(tmp_path):
    return parse_input(CHABOCHE)


@pytest.fixture
def simple(tmp_path):
    """A minimal hand-crafted input file written to a temp file."""
    content = textwrap.dedent(
        """\
        [Models]
          [elastic]
            type = LinearIsotropicElasticity
            youngs_modulus = 1e5
            poisson_ratio = 0.3
          []
          [yield]
            type = YieldFunction
            yield_stress = 'params/ys'
          []
        []
    """,
    )
    p = tmp_path / "simple.i"
    p.write_text(content)
    return parse_input(p)


# ---------------------------------------------------------------------------
# Section keys
# ---------------------------------------------------------------------------


def test_chaboche_sections(chaboche):
    assert set(chaboche.keys()) == {"Tensors", "Drivers", "Models", "EquationSystems", "Solvers"}


def test_simple_sections(simple):
    assert list(simple.keys()) == ["Models"]


# ---------------------------------------------------------------------------
# ModelEntry counts
# ---------------------------------------------------------------------------


def test_chaboche_models_count(chaboche):
    assert len(chaboche["Models"]) == 22


def test_simple_models_count(simple):
    assert len(simple["Models"]) == 2


def test_chaboche_tensors_count(chaboche):
    assert len(chaboche["Tensors"]) == 7


# ---------------------------------------------------------------------------
# ModelEntry fields
# ---------------------------------------------------------------------------


def test_first_model_name_and_type(chaboche):
    first = chaboche["Models"][0]
    assert first.name == "isoharden"
    assert first.type == "VoceIsotropicHardening"


def test_first_model_params(chaboche):
    first = chaboche["Models"][0]
    assert first.params["type"] == "VoceIsotropicHardening"
    assert first.params["saturated_hardening"] == "100"
    assert first.params["saturation_rate"] == "1.2"


def test_simple_entry_fields(simple):
    elastic = simple["Models"][0]
    assert elastic.name == "elastic"
    assert elastic.type == "LinearIsotropicElasticity"
    assert elastic.params["youngs_modulus"] == "1e5"
    assert elastic.params["poisson_ratio"] == "0.3"


# ---------------------------------------------------------------------------
# Value stripping
# ---------------------------------------------------------------------------


def test_quoted_value_stripped(simple):
    yield_entry = simple["Models"][1]
    # Single quotes around 'params/ys' should be stripped
    assert yield_entry.params["yield_stress"] == "params/ys"


def test_quoted_value_with_spaces(chaboche):
    # LinspaceSR2 has values = 'exx eyy ezz'
    tensor = next(e for e in chaboche["Tensors"] if e.name == "max_strain")
    assert tensor.params["values"] == "exx eyy ezz"


def test_inline_comment_stripped(tmp_path):
    content = textwrap.dedent(
        """\
        [Models]
          [m]
            type = Foo
            key = 42  # an inline comment
          []
        []
    """,
    )
    p = tmp_path / "comment.i"
    p.write_text(content)
    result = parse_input(p)
    assert result["Models"][0].params["key"] == "42"


# ---------------------------------------------------------------------------
# Duplicate section merging
# ---------------------------------------------------------------------------


def test_duplicate_sections_merged(tmp_path):
    content = textwrap.dedent(
        """\
        [Models]
          [a]
            type = Foo
          []
        []
        [Models]
          [b]
            type = Bar
          []
        []
    """,
    )
    p = tmp_path / "dup.i"
    p.write_text(content)
    result = parse_input(p)
    assert len(result["Models"]) == 2
    assert result["Models"][0].name == "a"
    assert result["Models"][1].name == "b"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_unmatched_close_raises(tmp_path):
    content = "[]  # nothing open\n"
    p = tmp_path / "bad.i"
    p.write_text(content)
    with pytest.raises(ValueError, match="Unmatched"):
        parse_input(p)


def test_unclosed_section_raises(tmp_path):
    content = "[Models]\n  [m]\n    type = Foo\n"
    p = tmp_path / "unclosed.i"
    p.write_text(content)
    with pytest.raises(ValueError, match="Unclosed"):
        parse_input(p)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_input("/nonexistent/path/model.i")


# ---------------------------------------------------------------------------
# ModelEntry is a dataclass
# ---------------------------------------------------------------------------


def test_model_entry_is_dataclass(simple):
    entry = simple["Models"][0]
    assert isinstance(entry, ModelEntry)
    assert hasattr(entry, "name")
    assert hasattr(entry, "type")
    assert hasattr(entry, "params")
    assert hasattr(entry, "children")
