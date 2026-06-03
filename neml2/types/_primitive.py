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

"""Generic intermediate layer for fixed-base-shape typed-tensor wrappers.

`PrimitiveTensor` sits between :class:`TensorWrapper` (purely abstract — shape
decomposition + region views + ``_rewrap``/``to``) and each concrete leaf
(``Scalar``, ``Vec``, ``R2``, ``SR2``, ``WR2``, ``Rot``, ``SSR4``,
``MillerIndex``). It provides three pieces of machinery that every primitive
needs in nearly-identical form:

1. **Generic arithmetic operators** routed through a single ``_binary``
   helper that handles the (Self × Self) and (Self × Scalar) dispatch
   cases. Each concrete primitive inherits ``+``, ``-``, ``*``, ``/``,
   unary ``-``, and the reflected variants for free. Cross-type ops
   (``R2.__matmul__``, ``Scalar * SR2 → SR2`` promotion) stay
   overridden on the leaves where they belong.
2. **Generic shape factories** — ``zeros`` / ``ones`` / ``full`` /
   ``empty`` — keyed on ``cls.BASE_SHAPE``. Each takes a dynamic batch
   shape as ``*batch`` positional args and pads the call to torch with
   the primitive's compile-time base shape.
3. **Generic ``fill`` template** — accept exactly ``prod(BASE_SHAPE)``
   scalar components and reshape into the base. ``SR2`` overrides this
   to add Mandel-aware overloads (1 / 3 / 6 component forms with √2
   shear scaling).

Mirrors the v2 C++ ``PrimitiveTensor<Derived, S...>`` CRTP template at
``include/neml2/tensors/PrimitiveTensor.h``. The Python equivalent uses
inheritance + ``Self`` typing instead of C++ macros — Python doesn't need
the macros, but the layering and naming carry over.

``PrimitiveTensor`` is not itself a ``@dataclass``; the leaves are
``@dataclass(frozen=True, eq=False)`` and pick up ``data: torch.Tensor``
plus ``sub_batch_ndim: int = 0`` as their concrete fields.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, ClassVar

import torch

from neml2.types._base import TensorWrapper, align_scalar_base, align_sub_batch

if TYPE_CHECKING:
    from typing_extensions import Self


class PrimitiveTensor(TensorWrapper):
    """Fixed-base-shape typed tensor with inherited arithmetic and factories."""

    # ``BASE_NDIM`` / ``BASE_SHAPE`` are inherited as ``ClassVar`` annotations
    # from ``TensorWrapper``; each concrete leaf supplies the values.

    # Lazy hook so the ``Self × Scalar`` branch in ``_binary`` can identify
    # Scalar without an import cycle. Populated by ``scalar.py`` at module
    # load time. Stays ``None`` if ``Scalar`` hasn't been imported yet — in
    # which case the Scalar branch is skipped and the op falls through to
    # ``NotImplemented`` (matching the pre-refactor behavior on non-Scalar
    # inputs).
    _SCALAR_CLS: ClassVar[type[TensorWrapper] | None] = None

    # ---- generic arithmetic dispatch ----

    def _binary(self: Self, other, op_fn) -> Self:
        """Dispatch a binary op between ``self`` and ``other``.

        Handles the two cases every primitive needs uniformly:

        1. ``other`` is the same concrete type: align sub-batch, apply ``op_fn``
           to the underlying tensors, re-wrap.
        2. ``other`` is a ``Scalar``: align sub-batch, broadcast the scalar's
           data against the wrapper's base shape via :func:`align_scalar_base`,
           apply ``op_fn``, re-wrap.

        Cross-type cases (e.g. ``R2 @ SR2``) and primitive-times-literal
        (``Scalar + 1.5``) fall through to ``NotImplemented`` — the calling
        leaf is expected to override the op and handle them explicitly.
        """
        if isinstance(other, type(self)):
            [aa, bb], sb = align_sub_batch(self, other)
            return type(self)(op_fn(aa.data, bb.data), sub_batch_ndim=sb)
        scalar_cls = type(self)._SCALAR_CLS
        if scalar_cls is not None and isinstance(other, scalar_cls):
            [aa, bb], sb = align_sub_batch(self, other)
            return type(self)(
                op_fn(aa.data, align_scalar_base(bb.data, type(self).BASE_NDIM)),
                sub_batch_ndim=sb,
            )
        return NotImplemented

    def _scale(self: Self, other, op_fn) -> Self:
        """Dispatch a scaling op (``*`` / ``/``) — same as :meth:`_binary` plus
        a ``float`` / ``int`` literal branch. Used by ``__mul__`` and
        ``__truediv__`` where component-wise scaling by a Python number has
        unambiguous meaning. ``__add__`` / ``__sub__`` deliberately do *not*
        accept literals: a uniform additive offset is rare and ambiguous
        enough to require an explicit ``cls.full(*batch, fill_value=...)``.
        """
        if isinstance(other, (float, int)):
            return type(self)(op_fn(self.data, other), sub_batch_ndim=self.sub_batch_ndim)
        return self._binary(other, op_fn)

    def __add__(self: Self, other) -> Self:
        return self._binary(other, operator.add)

    def __sub__(self: Self, other) -> Self:
        return self._binary(other, operator.sub)

    def __mul__(self: Self, other) -> Self:
        return self._scale(other, operator.mul)

    def __truediv__(self: Self, other) -> Self:
        return self._scale(other, operator.truediv)

    def __radd__(self: Self, other) -> Self:
        return self.__add__(other)

    def __rmul__(self: Self, other) -> Self:
        return self.__mul__(other)

    def __rsub__(self: Self, other) -> Self:
        # ``other - self``. Only meaningful if ``other`` is a same-type wrapper
        # or a Scalar, both of which would have defined their own ``__sub__``
        # already — so the only path we'd land here is when the LHS's
        # ``__sub__`` returned ``NotImplemented``. That's the cross-type case
        # the leaves handle, not this generic helper.
        return NotImplemented

    def __neg__(self: Self) -> Self:
        return type(self)(-self.data, sub_batch_ndim=self.sub_batch_ndim)

    # ---- generic shape factories ----

    @classmethod
    def zeros(
        cls,
        *batch: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        """Zero-filled wrapper of dynamic shape ``batch`` and base ``cls.BASE_SHAPE``."""
        return cls(torch.zeros(*batch, *cls.BASE_SHAPE, dtype=dtype, device=device))

    @classmethod
    def ones(
        cls,
        *batch: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        """Ones-filled wrapper of dynamic shape ``batch`` and base ``cls.BASE_SHAPE``."""
        return cls(torch.ones(*batch, *cls.BASE_SHAPE, dtype=dtype, device=device))

    @classmethod
    def full(
        cls,
        *batch: int,
        fill_value: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        """Wrapper of given shape filled with ``fill_value``."""
        return cls(torch.full((*batch, *cls.BASE_SHAPE), fill_value, dtype=dtype, device=device))

    @classmethod
    def empty(
        cls,
        *batch: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        """Wrapper of given shape with undefined data."""
        return cls(torch.empty(*batch, *cls.BASE_SHAPE, dtype=dtype, device=device))

    # ---- generic fill template ----

    @classmethod
    def fill(
        cls,
        *components: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        """Build a wrapper from ``prod(BASE_SHAPE)`` scalar components.

        Components are taken in row-major order and reshaped to ``cls.BASE_SHAPE``.
        Subclasses with non-trivial packing semantics (e.g. ``SR2`` with Mandel
        √2 shear scaling, or short-form 1/3-component overloads) override this.
        """
        expected = math.prod(cls.BASE_SHAPE) if cls.BASE_SHAPE else 1
        if len(components) != expected:
            raise ValueError(
                f"{cls.__name__}.fill expects {expected} components "
                f"(prod of BASE_SHAPE={cls.BASE_SHAPE}), got {len(components)}"
            )
        flat = torch.tensor(components, dtype=dtype or torch.float64, device=device)
        return cls(flat.reshape(cls.BASE_SHAPE) if cls.BASE_SHAPE else flat.squeeze())
