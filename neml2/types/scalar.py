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

"""Scalar — a physically-meaningful 0-base-shape tensor.

`Scalar` is a wrapper (not a `torch.Tensor` alias) so cross-type operator
dispatch (`Scalar * SR2 → SR2`) is deterministic via Python's reflected-operator
protocol. With a bare `torch.Tensor` on the left, Python would never invoke
`SR2.__rmul__`.

What Scalar overrides on top of :class:`PrimitiveTensor`:

- **Literal-friendly ``__init__``**: ``Scalar(2.5)`` and ``Scalar([1, 2, 3])``
  work directly; literals are coerced to ``torch.float64`` (the default
  precision for typed-wrapper algebra).
- **``float64`` factory defaults**: ``Scalar.zeros(n)``, ``Scalar.ones(n)``,
  ``Scalar.full(n, fill_value=...)`` default to ``torch.float64`` rather than
  torch's global ``float32`` default.
- **``linspace`` / ``arange``**: Scalar-only, mirroring the torch creation API.
- **``+`` / ``-`` with Python literals**: ``s + 1.5`` and ``s - 1`` are valid;
  the other primitives reject this (a uniform additive offset is rare and
  ambiguous on a (3,3) or (6,) tensor). Multiply / divide by literal are
  inherited from :meth:`PrimitiveTensor._scale`.
- **``__rsub__`` / ``__rtruediv__``** for the reverse-with-literal cases.
- **``__pow__``, ``__abs__``**: torch-backed forwarders, Scalar-only.

Everything else (``__neg__``, region views, ``zeros``/``ones`` shape semantics)
comes from the base layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeVar, overload

import torch

from neml2.types._base import TensorWrapper, align_sub_batch
from neml2.types._primitive import PrimitiveTensor
from neml2.types._pytree import register

_WrapperT = TypeVar("_WrapperT", bound=TensorWrapper)


@dataclass(frozen=True, eq=False)
class Scalar(PrimitiveTensor):
    """Wraps a `torch.Tensor` of base shape ``()`` (i.e., one number per batch entry)."""

    data: torch.Tensor
    sub_batch_ndim: int = 0
    BASE_NDIM: ClassVar[int] = 0
    BASE_SHAPE: ClassVar[tuple[int, ...]] = ()

    def __init__(
        self,
        data,
        sub_batch_ndim: int = 0,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if isinstance(data, torch.Tensor):
            if dtype is not None or device is not None:
                data = data.to(dtype=dtype, device=device)
        else:
            data = torch.as_tensor(data, dtype=dtype or torch.float64, device=device)
        # The class is `@dataclass(frozen=True)` so direct attribute writes
        # are forbidden; route through `object.__setattr__` to seat the
        # dataclass-declared fields.
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sub_batch_ndim", sub_batch_ndim)

    @classmethod
    def from_value(cls, x: float | int, *, like: TensorWrapper) -> Scalar:
        """Construct a Scalar inheriting dtype/device from an existing wrapper."""
        return cls(x, dtype=like.dtype, device=like.device)

    # ---- torch-analogue factories with float64 defaults ----
    #
    # Override the inherited PrimitiveTensor factories so Scalars default to
    # ``torch.float64`` (matching ``Scalar.__init__``) rather than torch's
    # global float32 default. Sub-batch tagging is composed post-hoc via the
    # fluent ``Scalar.linspace(...).sub_batch.retag(n)``.

    @classmethod
    def zeros(
        cls,
        *shape: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Scalar:
        return cls(torch.zeros(*shape, dtype=dtype or torch.float64, device=device))

    @classmethod
    def ones(
        cls,
        *shape: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Scalar:
        return cls(torch.ones(*shape, dtype=dtype or torch.float64, device=device))

    @classmethod
    def full(
        cls,
        *shape: int,
        fill_value: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Scalar:
        return cls(torch.full(shape, fill_value, dtype=dtype or torch.float64, device=device))

    @classmethod
    def linspace(
        cls,
        start: float,
        end: float,
        steps: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Scalar:
        """``steps`` values uniformly spaced from ``start`` to ``end`` inclusive."""
        return cls(torch.linspace(start, end, steps, dtype=dtype or torch.float64, device=device))

    @classmethod
    def arange(
        cls,
        start: float,
        end: float | None = None,
        step: float = 1,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Scalar:
        """Like ``torch.arange``: ``arange(N)`` -> ``[0, …, N-1]``, ``arange(a, b, s)``
        -> ``[a, a+s, …]`` up to (excluding) ``b``."""
        if end is None:
            return cls(torch.arange(start, dtype=dtype or torch.float64, device=device))
        return cls(torch.arange(start, end, step, dtype=dtype or torch.float64, device=device))

    # ---- arithmetic with Python literals (add/sub) ----
    #
    # ``__mul__`` / ``__truediv__`` are inherited via :meth:`PrimitiveTensor._scale`
    # which already handles ``float`` / ``int`` literals. ``__add__`` / ``__sub__``
    # need explicit overrides because the base ``_binary`` deliberately rejects
    # literals (a uniform additive offset is rare and ambiguous on a (3,3) or
    # (6,) tensor — on Scalar it's the natural thing).

    def __add__(self, other) -> Scalar:
        if isinstance(other, (float, int)):
            return Scalar(self.data + other, sub_batch_ndim=self.sub_batch_ndim)
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other) -> Scalar:
        return self.__add__(other)

    def __sub__(self, other) -> Scalar:
        if isinstance(other, (float, int)):
            return Scalar(self.data - other, sub_batch_ndim=self.sub_batch_ndim)
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other) -> Scalar:
        if isinstance(other, (float, int)):
            return Scalar(other - self.data, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented

    @overload
    def __mul__(self, other: Scalar | float | int) -> Scalar: ...
    @overload
    def __mul__(self, other: _WrapperT) -> _WrapperT: ...
    def __mul__(self, other):
        # The Scalar × Scalar and Scalar × literal cases are handled by the
        # inherited ``_scale``. The wrapper-promotion case (``Scalar * SR2 ->
        # SR2``) is delegated explicitly here so it type-checks at the
        # ``-> _WrapperT`` overload — without this branch we'd return
        # ``NotImplemented`` and rely on the wrapper's ``__rmul__``, which
        # works at runtime but isn't visible to type-checkers staring at the
        # ``Scalar * wrapper`` expression.
        if isinstance(other, TensorWrapper) and not isinstance(other, Scalar):
            return other * self
        return self._scale(other, lambda a, b: a * b)

    def __rmul__(self, other) -> Scalar:
        return self.__mul__(other)

    def __rtruediv__(self, other) -> Scalar:
        if isinstance(other, (float, int)):
            return Scalar(other / self.data, sub_batch_ndim=self.sub_batch_ndim)
        return NotImplemented

    # ---- Scalar-only transcendentals / unary ops ----
    #
    # ``__neg__`` is inherited from PrimitiveTensor — its body is the same.

    def __abs__(self) -> Scalar:
        return Scalar(torch.abs(self.data), sub_batch_ndim=self.sub_batch_ndim)

    def __pow__(self, n: float | int) -> Scalar:
        return Scalar(torch.pow(self.data, n), sub_batch_ndim=self.sub_batch_ndim)


register(Scalar)


# Tell PrimitiveTensor's generic ``_binary`` dispatch which class to recognise as
# Scalar for the (Self × Scalar) interop branch. This avoids an import cycle in
# ``_primitive.py`` (PrimitiveTensor doesn't know about Scalar at class-definition
# time, but every code path that triggers the Scalar branch must have already
# imported Scalar to construct one).
PrimitiveTensor._SCALAR_CLS = Scalar

# Unused import suppressed by being referenced here.
_ = align_sub_batch
