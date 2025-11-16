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

import pytest

from neml2 import VariableName


def test_ctors():
    A = VariableName()
    assert str(A) == ""

    A = VariableName("state")
    assert str(A) == "state"

    A = VariableName("force")
    assert str(A) == "force"

    A = VariableName("state/stress")
    assert str(A) == "state/stress"

    A = VariableName("state/internal/gamma")
    assert str(A) == "state/internal/gamma"


def test_empty():
    A = VariableName()
    B = VariableName("state/stress")
    assert not A
    assert B


def test_len():
    A = VariableName()
    B = VariableName("state/stress")
    assert len(A) == 0
    assert len(B) == 2


def test_with_suffix():
    A = VariableName("state/stress")
    B = VariableName("state/stress_foo")
    assert A.with_suffix("_foo") == B


def test_append():
    A = VariableName("state")
    B = VariableName("foo/bar")
    C = VariableName("state/foo/bar")
    assert A.append(B) == C


def test_prepend():
    A = VariableName("stress")
    B = VariableName("residual/stress")
    assert A.prepend(VariableName("residual")) == B


def test_start_with():
    A = VariableName("internal/stress/foo")
    B = VariableName("internal/stress")
    C = VariableName("residual/stress")
    assert A.start_with(B)
    assert not A.start_with(C)
