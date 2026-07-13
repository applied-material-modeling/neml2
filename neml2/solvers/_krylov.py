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

"""Shared plumbing for the matrix-free iterative (Krylov) linear solvers.

:class:`GMRES` / :class:`BiCGStab` are *linear* solvers used as a
:class:`~neml2.solvers.Newton` ``linear_solver`` dependency, exactly like
``DenseLU`` / ``SchurComplement``. Unlike those, they are **matrix-free**: they
never assemble ``A``, so they do not implement ``solve(A, b)``. Instead they are
recognized by ``is_iterative`` and expose :meth:`krylov_config`, and the eager
Newton (``neml2.solvers.newton``) + the AOTI exporter dispatch on that -- driving
the shared C++ Krylov loop over the compiled/eager ``Matvec`` (``J.v``) instead of
a baked direct solve.

The preconditioner and cache-strategy are two orthogonal axes (which ``M^-1``
approximates ``A^-1`` vs how often it is rebuilt across Newton iterations); both
are strings validated here to mirror the C++ ``parse_*`` in ``krylov.h``.
"""

from __future__ import annotations

_PRECONDITIONERS = ("none", "jacobi", "block_jacobi", "full")
_CACHE_STRATEGIES = ("none", "chord", "max_its")


def _validate_choice(value: str, allowed: tuple[str, ...], field: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {allowed}, got {value!r}")
    return value


class _KrylovSolver:
    """Common config + :meth:`krylov_config` for the iterative linear solvers.

    Concrete subclasses set :attr:`method` and provide the ``hit`` schema +
    ``from_hit``; this base provides the constructor, validation, and the config
    dict consumed by both the eager binding and the AOTI metadata.
    """

    SECTION = "Solvers"
    #: Marks a matrix-free iterative linear solver -- Newton + the AOTI exporter
    #: dispatch on this instead of calling ``solve(A, b)``.
    is_iterative = True
    #: Krylov method tag (``"gmres"`` / ``"bicgstab"``); overridden per subclass.
    method = "gmres"

    def __init__(
        self,
        *,
        restart: int = 40,
        max_its: int = 1000,
        abs_tol: float = 0.0,
        rel_tol: float = 1.0e-4,
        preconditioner: str = "none",
        cache_strategy: str = "none",
        cache_max_its: int = 10,
    ) -> None:
        self.restart = int(restart)
        self.max_its = int(max_its)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.preconditioner = _validate_choice(preconditioner, _PRECONDITIONERS, "preconditioner")
        self.cache_strategy = _validate_choice(cache_strategy, _CACHE_STRATEGIES, "cache_strategy")
        self.cache_max_its = int(cache_max_its)

    def krylov_config(self) -> dict:
        """Config forwarded to ``krylov_solve_eager`` and written to AOTI metadata.

        Keys match the ``krylov_solve_eager`` pybind kwargs and the C++
        ``KrylovConfig`` fields.
        """
        return {
            "method": self.method,
            "restart": self.restart,
            "max_its": self.max_its,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "preconditioner": self.preconditioner,
            "cache_strategy": self.cache_strategy,
            "cache_max_its": self.cache_max_its,
        }
