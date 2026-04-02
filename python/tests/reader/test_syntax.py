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

from neml2.reader._syntax import SyntaxDB, TypeInfo, ParamInfo

SYNTAX_CONTENT = textwrap.dedent(
    """\
    neml2::LinearIsotropicElasticity:
      section: Models
      doc: |-
        Relate elastic strain to stress for linear isotropic material.
        \\f$ \\boldsymbol{\\sigma} = \\lambda \\text{tr}(\\boldsymbol{\\varepsilon}) \\boldsymbol{I} + 2\\mu\\boldsymbol{\\varepsilon} \\f$
      _factory:
        type: neml2::Factory*
        ftype: NONE
        doc:
        suppressed: 1
        value: 0
      _settings:
        type: std::shared_ptr<neml2::Settings>
        ftype: NONE
        doc:
        suppressed: 1
        value: 0
      type:
        type: std::string
        ftype: NONE
        doc:
        suppressed: 0
        value: LinearIsotropicElasticity
      youngs_modulus:
        type: double
        ftype: PARAMETER
        doc: |-
          Young's modulus \\f$ E \\f$.
        suppressed: 0
        value:
      poisson_ratio:
        type: double
        ftype: PARAMETER
        doc: |-
          Poisson's ratio \\f$ \\nu \\f$.
        suppressed: 0
        value:
      strain:
        type: neml2::LabeledAxisAccessor
        ftype: INPUT
        doc: |-
          Mechanical strain input.
        suppressed: 0
        value: forces/E
      stress:
        type: neml2::LabeledAxisAccessor
        ftype: OUTPUT
        doc: |-
          Cauchy stress output.
        suppressed: 0
        value: state/S
    neml2::VoceIsotropicHardening:
      section: Models
      doc: |-
        Voce isotropic hardening model.
      saturated_hardening:
        type: double
        ftype: PARAMETER
        doc: |-
          Saturated isotropic hardening.
        suppressed: 0
        value:
      saturation_rate:
        type: double
        ftype: PARAMETER
        doc: |-
          Hardening saturation rate.
        suppressed: 0
        value:
""",
)


@pytest.fixture
def syntax_file(tmp_path):
    p = tmp_path / "syntax.yml"
    p.write_text(SYNTAX_CONTENT)
    return p


@pytest.fixture
def db(syntax_file):
    return SyntaxDB(syntax_file)


# ---------------------------------------------------------------------------
# available
# ---------------------------------------------------------------------------


def test_available(db):
    assert db.available is True


def test_unavailable(tmp_path):
    assert SyntaxDB(tmp_path / "nonexistent.yml").available is False


# ---------------------------------------------------------------------------
# lookup by short name
# ---------------------------------------------------------------------------


def test_lookup_short_name(db):
    info = db.lookup("LinearIsotropicElasticity")
    assert info is not None
    assert isinstance(info, TypeInfo)
    assert info.qualified_name == "neml2::LinearIsotropicElasticity"


def test_lookup_section(db):
    assert db.lookup("LinearIsotropicElasticity").section == "Models"


def test_lookup_doc_nonempty(db):
    assert len(db.lookup("LinearIsotropicElasticity").doc) > 0


def test_lookup_unknown_returns_none(db):
    assert db.lookup("NoSuchType") is None


# ---------------------------------------------------------------------------
# lookup by qualified name
# ---------------------------------------------------------------------------


def test_lookup_qualified_name(db):
    info = db.lookup("neml2::LinearIsotropicElasticity")
    assert info is not None
    assert info.qualified_name == "neml2::LinearIsotropicElasticity"


def test_lookup_qualified_unknown_returns_none(db):
    assert db.lookup("neml2::NoSuchType") is None


# ---------------------------------------------------------------------------
# params
# ---------------------------------------------------------------------------


def test_params_are_list_of_param_info(db):
    info = db.lookup("LinearIsotropicElasticity")
    assert all(isinstance(p, ParamInfo) for p in info.params)


def test_suppressed_params_excluded(db):
    names = [p.name for p in db.lookup("LinearIsotropicElasticity").params]
    assert "_factory" not in names
    assert "_settings" not in names


def test_type_param_excluded(db):
    names = [p.name for p in db.lookup("LinearIsotropicElasticity").params]
    assert "type" not in names


def test_param_ftypes(db):
    info = db.lookup("LinearIsotropicElasticity")
    by_name = {p.name: p for p in info.params}
    assert by_name["youngs_modulus"].ftype == "PARAMETER"
    assert by_name["strain"].ftype == "INPUT"
    assert by_name["stress"].ftype == "OUTPUT"


def test_param_doc(db):
    info = db.lookup("LinearIsotropicElasticity")
    by_name = {p.name: p for p in info.params}
    assert "Young" in by_name["youngs_modulus"].doc


# ---------------------------------------------------------------------------
# LaTeX stripping
# ---------------------------------------------------------------------------


def test_latex_stripped_from_type_doc(db):
    info = db.lookup("LinearIsotropicElasticity")
    assert "\\f$" not in info.doc


def test_latex_stripped_from_param_doc(db):
    info = db.lookup("LinearIsotropicElasticity")
    by_name = {p.name: p for p in info.params}
    assert "\\f$" not in by_name["youngs_modulus"].doc


# ---------------------------------------------------------------------------
# Unavailable DB warns and returns None
# ---------------------------------------------------------------------------


def test_unavailable_lookup_warns(tmp_path):
    bad_db = SyntaxDB(tmp_path / "missing.yml")
    with pytest.warns(UserWarning, match="syntax database"):
        result = bad_db.lookup("AnyType")
    assert result is None
