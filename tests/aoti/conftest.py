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

"""Register the shared test-only native models for the AOTI suite.

The ``request_ad_forward`` scenario uses ``SurrogateFlowRate`` (a request_AD
machine-learning surrogate), which lives in ``tests/regression/_fixtures`` rather
than the production package. Side-effect-import it so ``@register_neml2_object``
fires before the AOTI scenarios are collected -- the same mechanism
``tests/regression/conftest.py`` uses for the eager regression suite.
"""

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_REGRESSION = Path(__file__).resolve().parent.parent / "regression"
if str(_REGRESSION) not in sys.path:
    sys.path.insert(0, str(_REGRESSION))

# Side-effect import: ``@register_neml2_object`` fires for the test fixtures.
# ``import_module`` (rather than ``import _fixtures``) honors the dynamic sys.path
# entry above without pyright flagging an unresolved import (``_fixtures`` lives
# under tests/regression/, not this directory).
importlib.import_module("_fixtures")


@pytest.fixture(autouse=True)
def _isolate_aoti_extract_temp(monkeypatch):
    """Give each AOTI test its own extraction temp.

    torch's AOTIModelPackageLoader extracts a ``.pt2`` to a path under the
    *system* temp keyed by the artifact's **content hash**, and re-extracts on
    every load. Two tests whose compiled graphs are byte-identical -- e.g. models
    differing only in runtime solver config, whose residual/step ``.pt2`` are the
    same -- therefore share one extraction dir, which collides two ways:

    * **Windows (any run):** the first test's loaded ``.pyd`` stays locked for the
      process lifetime, so the second's extraction fails with a sharing violation
      ("error code: 32 ... file open failed").
    * **Parallel run (``pytest -n auto``, any OS):** two xdist workers extract
      into the same content-hash dir concurrently and one reads a half-written
      file -- surfacing as torch's own nlohmann ``parse error ... empty input`` on
      the ``.pt2``'s metadata.

    The Windows *within-a-single-test* variant of the collision (one test loads
    two byte-identical model variants -- e.g. a direct vs. iterative sensitivity
    solve of the same physics -- so the two ``.pyd`` extract to the same path and
    the second sharing-violates) is handled at the C++ loader instead: ``make_loader``
    (``neml2/csrc/aoti/Model.cpp``) gives every ``.pt2`` load its own unique temp
    dir on Windows, which a per-*test* fixture cannot. This fixture stays for the
    cross-test + parallel (POSIX metadata-race) isolation above and for cleanup.

    Point the system temp at a per-test scratch dir so each test (and each worker)
    extracts in isolation; torch reads the temp dir fresh per extraction, so the
    override takes effect. Use a SHORT ``mkdtemp`` under the real system temp, NOT
    the deeply-nested pytest ``tmp_path``: Python's multiprocessing puts its
    AF_UNIX socket under ``$TMPDIR`` too, and a long ``tmp_path`` blows the
    ~108-char socket limit (``OSError: AF_UNIX path too long``) in tests that use a
    process pool (e.g. the multidevice parallel-compile grid). Skipped for a plain
    sequential run off Windows, where re-extracting over a loaded ``.so`` is
    permitted and the shared cache is a harmless speedup.
    """
    if sys.platform != "win32" and not os.environ.get("PYTEST_XDIST_WORKER"):
        yield
        return
    extract = tempfile.mkdtemp(prefix="nml_aoti_")  # short, unique per test/worker
    # torch/tempfile honors TMPDIR on POSIX and TMP/TEMP on Windows.
    for var in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(var, extract)
    try:
        yield
    finally:
        shutil.rmtree(extract, ignore_errors=True)
