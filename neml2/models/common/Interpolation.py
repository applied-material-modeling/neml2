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

"""Python-native mirror of the C++ abstract ``Interpolation<T>`` base.

The C++ ``Interpolation`` (see ``include/neml2/models/common/Interpolation.h``)
is an abstract intermediate templated on the ordinate type ``T``. It declares
the shared "abscissa" / "ordinate" parameters, the "argument" input, and an
output that defaults to the model's block name. Concrete instantiations
(``ScalarLinearInterpolation``, ``BilinearInterpolation`` variants,
``HermiteSmoothStep``, ``SymmetricHermiteInterpolation``) are already ported
as standalone native leaves; this base only documents the shared contract.

Because the C++ template itself is not registered (no ``register_NEML2_object``
on the abstract base), this native port is also unregistered:
``@register_neml2_object`` is intentionally omitted. Native models are flat (no
schema inheritance), so concrete leaves declare their own ``HitSchema``
independently rather than inheriting from this base; ``forward`` raises
``NotImplementedError`` so any accidental direct use surfaces immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...schema import BLOCK_NAME, HitSchema, input, output, parameter
from ...types import Scalar
from ..chain_rule import ChainRuleDict
from ..model import Model

if TYPE_CHECKING:
    pass


class Interpolation(Model):
    """The base class for interpolated variable.

    This model requires two parameters, namely the "abscissa" and the
    "ordinate". The ordinate is interpolated using an input (specified by
    the "argument" option) along the axis of abscissa.

    The interpolant's batch shape is defined as the broadcasted batch shapes
    of the abscissa and the ordinate, after excluding the dimensions on
    which the interpolation happens.

    The general expectations for the batch shapes are:
    1. The abscissa and the ordinate should be batch-broadcastable.
    2. The abscissa should always be a Scalar. The ordinate can be of any
       primitive tensor type.
    3. The input (specified by option "argument") must be a Scalar.
    4. The input and the interpolant should be batch-broadcastable.
    5. Broadcasting the input with the interpolant should not alter its
       batch shape.
    """

    # Concrete native leaves declare their own ordinate-typed parameter and
    # output; the shared surface here is the Scalar "argument" input, the
    # block-name-defaulted output, and the Scalar "abscissa" parameter.
    hit = HitSchema(
        input("argument", Scalar, "Argument used to query the interpolant", attr="_argument"),
        output(
            "output",
            Scalar,
            "Output of the interpolant. If not specified, "
            "the object name will be used as the output name.",
            default=BLOCK_NAME,
            attr="_output",
        ),
        parameter("abscissa", Scalar, "Scalar defining the abscissa values of the interpolant"),
    )

    abscissa: Scalar
    _argument: str
    _output: str

    def forward(  # type: ignore[override]
        self,
        *inputs: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Abstract: concrete native interpolation leaves implement the
        # forward operator (argument -> interpolated value) and its
        # differential pushforward.
        raise NotImplementedError(
            f"{type(self).__name__} is an abstract base; subclass and implement forward()."
        )
