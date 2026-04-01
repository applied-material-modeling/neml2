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

import pytest

yaml = pytest.importorskip("yaml")

from neml2.reader._parser import parse_input
from neml2.reader._syntax import SyntaxDB
from neml2.reader._prompt import build_prompt
from neml2.reader import describe, explain

SYNTAX_CONTENT = textwrap.dedent(
    """\
    neml2::LinearIsotropicElasticity:
      section: Models
      doc: |-
        Relate elastic strain to stress for linear isotropic material.
      youngs_modulus:
        type: double
        ftype: PARAMETER
        doc: Young's modulus.
        suppressed: 0
        value:
      poisson_ratio:
        type: double
        ftype: PARAMETER
        doc: Poisson's ratio.
        suppressed: 0
        value:
"""
)

INPUT_CONTENT = textwrap.dedent(
    """\
    [Models]
      [elastic]
        type = LinearIsotropicElasticity
        youngs_modulus = 1e5
        poisson_ratio = 0.3
      []
      [unknown_model]
        type = SomeUnknownType
        param = value
      []
    []
"""
)


@pytest.fixture
def syntax_file(tmp_path):
    p = tmp_path / "syntax.yml"
    p.write_text(SYNTAX_CONTENT)
    return p


@pytest.fixture
def input_file(tmp_path):
    p = tmp_path / "model.i"
    p.write_text(INPUT_CONTENT)
    return p


@pytest.fixture
def db(syntax_file):
    return SyntaxDB(syntax_file)


@pytest.fixture
def parsed(input_file):
    return parse_input(input_file)


# ---------------------------------------------------------------------------
# build_prompt structure
# ---------------------------------------------------------------------------


def test_returns_two_strings(parsed, db):
    result = build_prompt(parsed, db)
    assert isinstance(result, tuple)
    assert len(result) == 2
    system, user = result
    assert isinstance(system, str)
    assert isinstance(user, str)


def test_system_prompt_nonempty(parsed, db):
    system, _ = build_prompt(parsed, db)
    assert len(system) > 0


def test_user_prompt_contains_model_name(parsed, db):
    _, user = build_prompt(parsed, db)
    assert "elastic" in user


def test_user_prompt_contains_type(parsed, db):
    _, user = build_prompt(parsed, db)
    assert "LinearIsotropicElasticity" in user


def test_user_prompt_contains_description(parsed, db):
    _, user = build_prompt(parsed, db)
    assert "linear isotropic" in user.lower()


def test_user_prompt_contains_params(parsed, db):
    _, user = build_prompt(parsed, db)
    assert "youngs_modulus" in user
    assert "1e5" in user


def test_user_prompt_contains_param_doc(parsed, db):
    _, user = build_prompt(parsed, db)
    assert "Young's modulus" in user


def test_unknown_type_noted(parsed, db):
    _, user = build_prompt(parsed, db)
    assert "unknown_model" in user
    assert "not found" in user


# ---------------------------------------------------------------------------
# include_params=False
# ---------------------------------------------------------------------------


def test_include_params_false_omits_values(parsed, db):
    _, user = build_prompt(parsed, db, include_params=False)
    assert "1e5" not in user


def test_include_params_false_omits_param_doc(parsed, db):
    _, user = build_prompt(parsed, db, include_params=False)
    assert "Young's modulus" not in user


# ---------------------------------------------------------------------------
# system_context
# ---------------------------------------------------------------------------


def test_system_context_appended(parsed, db):
    ctx = "The user is a structural engineer."
    system, _ = build_prompt(parsed, db, system_context=ctx)
    assert ctx in system


# ---------------------------------------------------------------------------
# unavailable DB
# ---------------------------------------------------------------------------


def test_unavailable_db_warns_and_continues(parsed, tmp_path):
    bad_db = SyntaxDB(tmp_path / "missing.yml")
    with pytest.warns(UserWarning):
        system, user = build_prompt(parsed, bad_db)
    assert "elastic" in user
    assert "Syntax database not available" in user


# ---------------------------------------------------------------------------
# describe() public API
# ---------------------------------------------------------------------------


def test_describe_returns_tuple(input_file, syntax_file):
    result = describe(input_file, syntax_file)
    assert isinstance(result, tuple) and len(result) == 2


def test_describe_missing_file_raises(syntax_file):
    with pytest.raises(FileNotFoundError):
        describe("/nonexistent/model.i", syntax_file)


# ---------------------------------------------------------------------------
# explain() public API (mock client)
# ---------------------------------------------------------------------------


class MockClient:
    def __init__(self, response="Mock explanation."):
        self.response = response
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.response


def test_explain_calls_client(input_file, syntax_file):
    client = MockClient("This is an elastic model.")
    result = explain(input_file, syntax_file, client)
    assert result == "This is an elastic model."
    assert len(client.calls) == 1


def test_explain_passes_system_and_user(input_file, syntax_file):
    client = MockClient()
    explain(input_file, syntax_file, client)
    system, user = client.calls[0]
    assert "NEML2" in system
    assert "elastic" in user


def test_explain_system_context(input_file, syntax_file):
    client = MockClient()
    ctx = "Extra context."
    explain(input_file, syntax_file, client, system_context=ctx)
    system, _ = client.calls[0]
    assert ctx in system
