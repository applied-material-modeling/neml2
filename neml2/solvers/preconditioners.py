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

"""Preconditioners for the matrix-free iterative (Krylov) linear solvers.

A preconditioner is a registered ``[Solvers]`` object taken by ``GMRES`` /
``BiCGStab`` as a dependency (default :class:`NoPreconditioner`). It authors a
pair of modules:

- ``setup_module(system)`` -- ``(*u, *g, *params) -> (*state)``: build the
  preconditioner state (factored / inverted from the model), compiled to a
  ``_precond_setup`` graph (or run live for eager). The C++ Krylov runtime holds
  the returned state and rebuilds it per the ``cache_strategy``.
- ``apply_module(system)`` -- ``(*state, r_flat) -> z_flat``: apply ``M^-1`` to
  the flat residual, compiled to ``_precond_apply`` (or run live).

Because both are ordinary authored torch modules, a user can add a custom
preconditioner by subclassing :class:`Preconditioner` and returning their own
setup/apply modules -- no C++ change. v1 targets a single dense unknown group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neml2.factory import register_neml2_object
from neml2.schema import HitSchema

if TYPE_CHECKING:
    import nmhit
    from torch import nn

    from neml2.es import ModelNonlinearSystem
    from neml2.factory import _NativeInputFile


def _unknown_block_sizes(system: ModelNonlinearSystem) -> tuple[int, ...]:
    """Per-variable storage widths of the (single dense) unknown group -- the
    Block-Jacobi diagonal block sizes (summing to the flat unknown count)."""
    from neml2.es.assembled import AssembledMatrix  # noqa: PLC0415

    lay = system.ulayout
    return tuple(
        AssembledMatrix._var_storage(lay, gi, name, lay.structure[gi])
        for gi in range(lay.ngroup)
        for name in lay.groups[gi]
    )


class Preconditioner:
    """Base preconditioner. Concrete subclasses set :attr:`kind` and return their
    setup/apply modules; the base is the ``none`` (identity) case."""

    SECTION = "Solvers"
    #: Marks a preconditioner object (GMRES/BiCGStab + the exporter dispatch on it).
    is_preconditioner = True
    #: Metadata tag (informational; the behavior is defined by the graphs).
    kind = "none"

    def setup_module(self, system: ModelNonlinearSystem) -> nn.Module | None:
        """The setup module, or ``None`` for the identity (no) preconditioner."""
        del system
        return None

    def apply_module(self, system: ModelNonlinearSystem) -> nn.Module | None:
        del system
        return None


@register_neml2_object("NoPreconditioner")
class NoPreconditioner(Preconditioner):
    """Identity preconditioner (unpreconditioned Krylov)."""

    kind = "none"
    hit = HitSchema()

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> NoPreconditioner:
        del node, factory
        return cls()


@register_neml2_object("JacobiPreconditioner")
class JacobiPreconditioner(Preconditioner):
    """Point-Jacobi: ``M = diag(A)`` (elementwise scaling)."""

    kind = "jacobi"
    hit = HitSchema()

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> JacobiPreconditioner:
        del node, factory
        return cls()

    def setup_module(self, system: ModelNonlinearSystem) -> nn.Module:
        from neml2.es.implicit import JacobiPrecondSetup  # noqa: PLC0415

        return JacobiPrecondSetup(system)

    def apply_module(self, system: ModelNonlinearSystem) -> nn.Module:
        from neml2.es.implicit import JacobiPrecondApply  # noqa: PLC0415

        del system
        return JacobiPrecondApply()


@register_neml2_object("BlockJacobiPreconditioner")
class BlockJacobiPreconditioner(Preconditioner):
    """Block-Jacobi: ``M = blockdiag(A)`` over the per-variable on-diagonal blocks."""

    kind = "block_jacobi"
    hit = HitSchema()

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> BlockJacobiPreconditioner:
        del node, factory
        return cls()

    def setup_module(self, system: ModelNonlinearSystem) -> nn.Module:
        from neml2.es.implicit import BlockJacobiPrecondSetup  # noqa: PLC0415

        return BlockJacobiPrecondSetup(system, _unknown_block_sizes(system))

    def apply_module(self, system: ModelNonlinearSystem) -> nn.Module:
        from neml2.es.implicit import BlockJacobiPrecondApply  # noqa: PLC0415

        return BlockJacobiPrecondApply(_unknown_block_sizes(system))


@register_neml2_object("FullPreconditioner")
class FullPreconditioner(Preconditioner):
    """Full (LU) preconditioner: ``M = A`` (exact per-solve; with a caching
    ``cache_strategy`` this is modified Newton -- factor once, reuse)."""

    kind = "full"
    hit = HitSchema()

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> FullPreconditioner:
        del node, factory
        return cls()

    def setup_module(self, system: ModelNonlinearSystem) -> nn.Module:
        from neml2.es.implicit import FullPrecondSetup  # noqa: PLC0415

        return FullPrecondSetup(system)

    def apply_module(self, system: ModelNonlinearSystem) -> nn.Module:
        from neml2.es.implicit import FullPrecondApply  # noqa: PLC0415

        del system
        return FullPrecondApply()


__all__ = [
    "Preconditioner",
    "NoPreconditioner",
    "JacobiPreconditioner",
    "BlockJacobiPreconditioner",
    "FullPreconditioner",
]
