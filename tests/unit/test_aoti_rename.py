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

"""Boundary-rename validation + CLI plumbing for ``neml2-compile``.

Fast, compile-free coverage of the shallow boundary-rename feature:

* :func:`neml2.cli.aoti_export._validate_renames` -- the fail-fast validator
  (unknown name, duplicate boundary name, non-promoted parameter, identity
  drop, unknown namespace).
* the ``neml2-compile`` CLI surface -- ``ORIG:NEW`` parsing and the
  ``--model``-only restriction (rejecting ``--rename-* + --driver``).

The end-to-end metadata + numerical round-trip (which requires a real AOTI
compile) lives in ``tests/aoti/test_rename.py``.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from neml2.cli.aoti_export import _validate_renames

# --------------------------------------------------------------------------- #
# _validate_renames                                                           #
# --------------------------------------------------------------------------- #

_IN = ["strain", "temperature"]
_OUT = ["stress"]
_PARAM = ["elasticity.E", "elasticity.nu"]


def _validate(renames):
    return _validate_renames(renames, input_names=_IN, output_names=_OUT, param_names=_PARAM)


def test_validate_renames_none_and_empty():
    empty = {"inputs": {}, "outputs": {}, "parameters": {}}
    assert _validate(None) == empty
    assert _validate({}) == empty


def test_validate_renames_normalizes_each_namespace():
    out = _validate(
        {
            "inputs": {"strain": "eps"},
            "outputs": {"stress": "sig"},
            "parameters": {"elasticity.E": "youngs"},
        }
    )
    assert out == {
        "inputs": {"strain": "eps"},
        "outputs": {"stress": "sig"},
        "parameters": {"elasticity.E": "youngs"},
    }


def test_validate_renames_drops_identity():
    # A no-op rename (boundary == original) is dropped so the metadata stays lean.
    assert _validate({"outputs": {"stress": "stress"}})["outputs"] == {}


def test_validate_renames_unknown_input_raises():
    with pytest.raises(ValueError, match=r"Renamed input name\(s\) not found.*nope"):
        _validate({"inputs": {"nope": "x"}})


def test_validate_renames_unknown_output_raises():
    with pytest.raises(ValueError, match=r"Renamed output name\(s\) not found"):
        _validate({"outputs": {"missing": "x"}})


def test_validate_renames_non_promoted_parameter_raises():
    # Only promoted-parameter qnames are valid rename targets.
    with pytest.raises(ValueError, match=r"Renamed parameter name\(s\) not found"):
        _validate({"parameters": {"elasticity.G": "shear"}})


def test_validate_renames_collision_with_sibling_raises():
    # Renaming one input onto another input's (unrenamed) name collides.
    with pytest.raises(ValueError, match=r"duplicate boundary name.*temperature"):
        _validate({"inputs": {"strain": "temperature"}})


def test_validate_renames_collision_between_two_new_names_raises():
    with pytest.raises(ValueError, match=r"duplicate boundary name"):
        _validate({"inputs": {"strain": "same", "temperature": "same"}})


def test_validate_renames_unknown_namespace_raises():
    with pytest.raises(ValueError, match=r"Unknown rename namespace"):
        _validate({"bogus": {"a": "b"}})


# --------------------------------------------------------------------------- #
# CLI parsing + --model-only restriction                                      #
# --------------------------------------------------------------------------- #


def test_cli_parse_rename_pairs():
    from neml2.cli.aoti_compile import _build_arg_parser, _renames_from_args

    args = _build_arg_parser().parse_args(
        [
            "x.i",
            "--model",
            "m",
            "--rename-input",
            "strain:eps",
            "--rename-output",
            "stress:sig",
            "--rename-parameter",
            "elasticity.E:youngs",
        ]
    )
    assert _renames_from_args(args) == {
        "inputs": {"strain": "eps"},
        "outputs": {"stress": "sig"},
        "parameters": {"elasticity.E": "youngs"},
    }


def test_cli_parse_rename_qualified_param_splits_on_first_colon():
    # A qualified promoted-parameter name uses dots; only the first ':' separates.
    from neml2.cli.aoti_compile import _parse_rename_pairs

    assert _parse_rename_pairs(["sub.model.E:youngs"], "--rename-parameter") == {
        "sub.model.E": "youngs"
    }


@pytest.mark.parametrize("spec", ["nocolon", "strain:", ":eps", ""])
def test_cli_parse_rename_malformed_raises(spec):
    from neml2.cli.aoti_compile import _parse_rename_pairs

    with pytest.raises(ValueError):
        _parse_rename_pairs([spec], "--rename-input")


def test_cli_parse_rename_duplicate_orig_raises():
    from neml2.cli.aoti_compile import _parse_rename_pairs

    with pytest.raises(ValueError, match=r"duplicate rename"):
        _parse_rename_pairs(["strain:a", "strain:b"], "--rename-input")


def _main_stderr(argv):
    """Run ``neml2-compile`` main, returning (SystemExit code, stderr)."""
    from neml2.cli.aoti_compile import main

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code, buf.getvalue()


def test_cli_rename_with_driver_errors():
    # Renaming is only supported in --model mode; --driver must be rejected before
    # any compile work (so the nonexistent input path is never reached).
    code, err = _main_stderr(["x.i", "--driver", "d", "--rename-input", "strain:eps"])
    assert code == 2
    assert "only supported with --model" in err


def test_cli_rename_malformed_token_errors():
    code, err = _main_stderr(["x.i", "--model", "m", "--rename-output", "bad_token"])
    assert code == 2
    assert "--rename-output" in err
