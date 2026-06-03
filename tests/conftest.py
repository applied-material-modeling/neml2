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
import torch

torch.set_default_dtype(torch.float64)


def pytest_addoption(parser):
    parser.addoption(
        "--run-aoti-compile",
        action="store_true",
        default=False,
        help="run tests that compile AOTI packages",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "aoti_compile: tests that compile AOTI packages and are skipped by default",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-aoti-compile"):
        return
    skip_aoti = pytest.mark.skip(reason="requires --run-aoti-compile")
    for item in items:
        if "aoti_compile" in item.keywords:
            item.add_marker(skip_aoti)
