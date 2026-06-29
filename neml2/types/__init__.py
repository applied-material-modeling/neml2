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

"""Typed Python tensor wrappers.

Each class is a `@dataclass(frozen=True, eq=False)` over a single
``data: torch.Tensor`` field, registered as a pytree node so `torch.export`
flattens cleanly at the export boundary while the authoring surface stays
strongly typed inside `nn.Module.forward()`.

Operator overloads (``+``, ``-``, ``*``, ``@``, ``-x``, ``abs(x)``, ``x ** n``)
live on the wrapper classes along with constructors. Shape-manipulation
ops are exposed through region-view properties (``t.batch``,
``t.dynamic_batch``, ``t.sub_batch``, ``t.base``) so the intent is
unambiguous — e.g. ``t.sub_batch.unsqueeze(-1)`` or
``t.base.transpose(-2, -1)``. Everything else (invariants,
decompositions, transcendentals, math-bearing type conversions like
``euler_rodrigues(Rot) -> R2``) lives in :mod:`neml2.types.functions`
as free functions, matching how the C++ side exposes them.
"""

from neml2.types._base import TensorWrapper, align_sub_batch
from neml2.types._primitive import PrimitiveTensor
from neml2.types.functions import (
    abs,
    allclose,
    bilinear_interpolation,
    bilinear_interpolation_slopes,
    cat,
    clamp,
    compose,
    cosh,
    det,
    dev,
    dexp_map,
    diff,
    drotate,
    drotate_self,
    equal,
    euler_rodrigues,
    exp,
    exp_map,
    gt,
    heaviside,
    inner,
    inv,
    jvp_compose,
    jvp_euler_rodrigues,
    jvp_exp_map,
    jvp_linear_interpolation,
    jvp_rotate,
    linear_interpolation,
    linspace,
    log,
    log10,
    logspace,
    lt,
    macaulay,
    mean,
    norm,
    opaque_pow,
    outer,
    pow,
    r2_from_sr2,
    r2_from_wr2,
    reciprocal,
    rotate,
    sign,
    sinh,
    skew,
    sqrt,
    stack,
    sum,
    sym,
    tanh,
    tr,
    unit,
    vec_component,
    vec_from_scalars,
    vol,
    where,
)
from neml2.types.miller_index import MillerIndex
from neml2.types.r2 import R2
from neml2.types.rot import Rot
from neml2.types.scalar import Scalar
from neml2.types.sr2 import SR2
from neml2.types.ssr4 import SSR4
from neml2.types.tensor import AxisKind, Tensor
from neml2.types.vec import Vec
from neml2.types.wr2 import WR2

__all__ = [
    "AxisKind",
    "MillerIndex",
    "PrimitiveTensor",
    "R2",
    "Rot",
    "SR2",
    "SSR4",
    "Scalar",
    "Tensor",
    "TensorWrapper",
    "Vec",
    "WR2",
    "abs",
    "align_sub_batch",
    "bilinear_interpolation",
    "bilinear_interpolation_slopes",
    "cat",
    "clamp",
    "compose",
    "cosh",
    "det",
    "dev",
    "dexp_map",
    "diff",
    "drotate",
    "drotate_self",
    "euler_rodrigues",
    "exp",
    "exp_map",
    "gt",
    "heaviside",
    "inner",
    "inv",
    "jvp_compose",
    "jvp_euler_rodrigues",
    "jvp_exp_map",
    "jvp_linear_interpolation",
    "jvp_rotate",
    "linear_interpolation",
    "linspace",
    "log",
    "log10",
    "logspace",
    "lt",
    "macaulay",
    "mean",
    "norm",
    "opaque_pow",
    "outer",
    "pow",
    "r2_from_sr2",
    "r2_from_wr2",
    "reciprocal",
    "rotate",
    "sinh",
    "skew",
    "sign",
    "sqrt",
    "stack",
    "sum",
    "sym",
    "tanh",
    "tr",
    "unit",
    "vec_component",
    "vec_from_scalars",
    "vol",
    "where",
]
