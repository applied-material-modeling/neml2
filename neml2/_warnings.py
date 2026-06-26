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

"""Scoped suppressors for benign torch-internal warnings.

torch raises these from its own internals, on code paths the caller cannot act on
(it reaches for its own deprecated APIs). We deliberately avoid a process-wide or
pytest-wide ``filterwarnings`` ignore: a blanket ignore of one of these messages
would also mask it if a *different*, genuine bug ever started raising the same
warning elsewhere. Instead each is silenced with a narrow :func:`ignore_warnings`
block wrapped around exactly the call that provokes it -- in the library for the
paths real users hit (the AOTI lowering in :mod:`neml2.models.export`) and in the
tests for the paths only the suite hits (a test calling ``torch.export`` directly,
a forward-mode ``torch.func.jvp`` in a fixture).

The ``(message, category)`` specs below are the single source of truth: reference
a constant from here rather than re-spelling a regex at the call site, so the
library and the tests can never drift. Each ``message`` is a regex matched against
the *start* of the warning text (``warnings`` semantics) and is pinned to a
category, so it cannot mask an unrelated warning.

(pyzag's process-wide ``donated_buffer`` notice fires from several pyzag-test
solver constructions rather than one call, so it is silenced once in
``pyproject.toml``'s ``filterwarnings`` instead -- still spelled in exactly one
place.)
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager

#: A ``(message-regex, category)`` pair, as consumed by :func:`warnings.filterwarnings`.
WarningSpec = tuple[str, type[Warning]]

#: torch reaches for its own deprecated ``torch.jit.script`` / ``torch.jit.script_method``
#: under Python 3.14 -- while importing ``torch.utils.mkldnn`` during an inductor
#: compile, and while registering forward-mode JVP decompositions the first time a
#: ``torch.func.jvp`` runs. neml2 calls neither API. Drop once torch supports
#: ``torch.jit`` on Python 3.14.
TORCH_JIT_PY314: WarningSpec = (
    r"`torch\.jit\.script(_method)?` is not supported in Python 3\.14",
    DeprecationWarning,
)

#: torch's own deprecated pytree call, raised during the pytree deepcopy inside
#: ``torch.export`` / ``run_decompositions``.
TORCH_TREESPEC_LEAFSPEC: WarningSpec = (
    r"`isinstance\(treespec, LeafSpec\)` is deprecated",
    FutureWarning,
)


@contextmanager
def ignore_warnings(*specs: WarningSpec) -> Generator[None, None, None]:
    """Suppress the given benign-warning ``specs`` for the duration of the block.

    Each spec is one of the module-level constants above. The filters are installed
    inside :func:`warnings.catch_warnings`, so they are removed on exit and never
    leak to surrounding code.
    """
    with warnings.catch_warnings():
        for message, category in specs:
            warnings.filterwarnings("ignore", message=message, category=category)
        yield
