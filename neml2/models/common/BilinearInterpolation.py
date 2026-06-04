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

"""Python-native mirrors of C++ ``common/BilinearInterpolation.h``.

The C++ header is templated on the ordinate type $T$ and the source file
registers three instantiations — ``Scalar``, ``Vec``, and ``SR2``. This
module ships the matching trio of native leaves, all delegating to the
typed :func:`~neml2.types.bilinear_interpolation` primitive (with the
analytical chain-rule slopes from
:func:`~neml2.types.bilinear_interpolation_slopes`).
"""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import BLOCK_NAME, HitSchema, input, output, parameter
from ...types import (
    SR2,
    Scalar,
    TensorWrapper,
    Vec,
    bilinear_interpolation,
    bilinear_interpolation_slopes,
)


def _make_bilinear(type_name: str, ordinate_cls: type[TensorWrapper]) -> type[Model]:
    """Build a ``{Type}BilinearInterpolation`` leaf bound to ``ordinate_cls``.

    Mirrors C++ ``BilinearInterpolation<T>`` for ``T in {Scalar, Vec, SR2}``.
    The ordinate parameter is a typed wrapper with ``sub_batch_ndim=2`` whose
    inner two sub-batch axes index the rectilinear cell; the abscissae are
    Scalars with ``sub_batch_ndim=1`` of sizes ``N1`` and ``N2``; the
    arguments are Scalars at the query point.
    """

    @register_neml2_object(type_name)
    class _BilinearInterpolation(Model):
        # Match the C++ ``Interpolation<T>`` HIT surface: ``ordinate`` is a
        # typed parameter, the output defaults to the model's block name.
        hit = HitSchema(
            input("argument1", Scalar, "First argument used to query the interpolant"),
            input("argument2", Scalar, "Second argument used to query the interpolant"),
            output(
                "output",
                ordinate_cls,
                "Output of the interpolant. If not specified, the object name will be used.",
                default=BLOCK_NAME,
                attr="_output",
            ),
            parameter(
                "abscissa1",
                Scalar,
                "Scalar defining the abscissa values of the first interpolation axis",
            ),
            parameter(
                "abscissa2",
                Scalar,
                "Scalar defining the abscissa values of the second interpolation axis",
            ),
            parameter(
                "ordinate",
                ordinate_cls,
                f"{ordinate_cls.__name__} defining the ordinate values of the interpolant",
            ),
        )

        # Class-level type hints so pyright sees the re-wrapped types from
        # ``Model.__getattr__`` (otherwise it infers ``Module`` from
        # ``nn.Module``'s registered-parameter / buffer accessor).
        abscissa1: Scalar
        abscissa2: Scalar
        _output: str

        def forward(  # type: ignore[override]
            self,
            x1: Scalar,
            x2: Scalar,
            *nl_params: TensorWrapper,
            v: ChainRuleDict | None = None,
        ):
            X1 = self.abscissa1
            X2 = self.abscissa2
            Y = self._get_param("ordinate", nl_params, ordinate_cls)
            # The ordinate as registered carries ``sub_batch_ndim=0``; re-tag
            # to the (N1, N2) sub-batch layout the primitive expects.
            Y_sb = Y.sub_batch.retag(2)
            X1_sb = X1.sub_batch.retag(1)
            X2_sb = X2.sub_batch.retag(1)
            out = bilinear_interpolation(x1, x2, X1_sb, X2_sb, Y_sb)
            if v is None:
                return out

            slope1, slope2 = bilinear_interpolation_slopes(x1, x2, X1_sb, X2_sb, Y_sb)
            slope1_c = slope1
            slope2_c = slope2

            def x1_action(V: Scalar) -> TensorWrapper:
                # slope * V broadcasts the Scalar tangent across the ordinate's
                # base dims (Vec/SR2 handle Scalar-on-the-right multiplication
                # by base-unsqueezing internally).
                return slope1_c * V  # type: ignore[operator]

            def x2_action(V: Scalar) -> TensorWrapper:
                return slope2_c * V  # type: ignore[operator]

            return out, self.apply_chain_rule(
                v,
                self._output,
                {"argument1": x1_action, "argument2": x2_action},
                output=out,
            )

    _BilinearInterpolation.__name__ = type_name
    _BilinearInterpolation.__qualname__ = type_name
    _BilinearInterpolation.__doc__ = (
        f"Bilinearly interpolate a {ordinate_cls.__name__} parameter on a 2-D grid.\n\n"
        "See ``neml2::Interpolation`` for the shape rules and "
        "``neml2::BilinearInterpolation`` for the corner formula."
    )
    return _BilinearInterpolation


ScalarBilinearInterpolation = _make_bilinear("ScalarBilinearInterpolation", Scalar)
VecBilinearInterpolation = _make_bilinear("VecBilinearInterpolation", Vec)
SR2BilinearInterpolation = _make_bilinear("SR2BilinearInterpolation", SR2)


__all__ = [
    "ScalarBilinearInterpolation",
    "VecBilinearInterpolation",
    "SR2BilinearInterpolation",
]
