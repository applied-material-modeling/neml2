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

"""Canonical solver exception surface, shared across every evaluation route.

``ConvergenceError`` signals a *recoverable* solve failure (a Newton
divergence / max-iterations), as opposed to a fatal shape/device/config bug --
a consumer (the substepping driver, MOOSE's outer time-stepper) can catch it,
cut the increment, and retry.

The type itself **lives in C++** (``neml2/csrc/aoti/Exception.h``,
``neml2::aoti::ConvergenceError``) and is surfaced to Python by the pybind
layer (``neml2/csrc/aoti/_aoti.cpp``). The shared C++ Newton -- which every
route drives, including the eager path via ``newton_solve_eager`` -- raises it
directly, so aliasing it here (rather than defining a parallel Python class)
means a single ``except neml2.solvers.ConvergenceError`` catches a divergence
no matter which route produced it. By the time this module is imported,
``neml2.aoti._aoti`` is already loaded (``neml2/__init__`` imports ``aoti``
before ``solvers``), so this is a symbol lookup, not a new import cycle.

The pure-Python fallback exists only for a partial build with no compiled
extension (where no C++ solve -- hence no ``ConvergenceError`` -- can occur);
it keeps ``import neml2.solvers`` working in that degenerate case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For the type checker, present a stable RuntimeError subclass. At runtime it
    # is aliased to the C++-registered type (below) so identity is unified.
    class ConvergenceError(RuntimeError):
        """Recoverable nonlinear-solve failure (Newton divergence / max-iters)."""
else:
    try:
        from neml2.aoti._aoti import ConvergenceError
    except ImportError:  # pragma: no cover - partial build without the C++ extension

        class ConvergenceError(RuntimeError):
            """Recoverable nonlinear-solve failure (pure-Python partial-build stand-in)."""


__all__ = ["ConvergenceError"]
