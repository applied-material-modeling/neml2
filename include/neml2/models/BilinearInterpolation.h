// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "neml2/models/Interpolation.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
/**
 * @brief Linearly interpolate the parameter along on a 2D grid.
 *
 * Currently, this object is hard-coded to always interpolate along the last two batch dimensions.
 * A few examples of tensor shapes are listed below to demonstrate how broadcasting is handled:
 *
 * Example 1: unbatched abscissa, unbatched ordinate (of type R2), unbatched input argument,
 * interpolation grid 6x8
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * abscissa 1 shape: (   6;     )
 * abscissa 2 shape: (   8;     )
 *   ordinate shape: (6, 8; 3, 3)
 *    input 1 shape: (    ;     )
 *    input 2 shape: (    ;     )
 *     output shape: (    ; 3, 3)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Example 2: unbatched abscissa, unbatched ordinate (of type R2), batched input argument (with
 * batch shape `(2, 3)`), interpolation grid 6x8
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * abscissa 1 shape: (         6;     )
 * abscissa 2 shape: (         8;     )
 *   ordinate shape: (      6, 8; 3, 3)
 *    input 1 shape: (2, 3      ;     )
 *    input 2 shape: (2, 3      ;     )
 *     output shape: (2, 3      ; 3, 3)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Example 3: unbatched abscissa, batched ordinate (of type R2 and with batch shape `(5, 1)`),
 * batched input argument (with batch shape `(2, 5, 2)`), interpolation grid 6x8
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * abscissa 1 shape: (            6;     )
 * abscissa 2 shape: (            8;     )
 *   ordinate shape: (   5, 1, 6, 8; 3, 3)
 *    input 1 shape: (2, 5, 2      ;     )
 *    input 2 shape: (2, 5, 2      ;     )
 *     output shape: (2, 5, 2      ; 3, 3)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Example 4: batched abscissa (with batch shape `(7, 8, 1)`), unbatched ordinate (of type R2),
 * batched input argument (with batch shape `(7, 1, 5)` and `(8, 1)`), interpolation grid 10x10
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * abscissa 1 shape: (            6;     )
 * abscissa 2 shape: (            8;     )
 *   ordinate shape: (         6, 8; 3, 3)
 *    input 1 shape: (7, 1, 5      ;     )
 *    input 2 shape: (   8, 1      ;     )
 *     output shape: (7, 8, 5      ; 3, 3)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 */
template <typename T>
class BilinearInterpolation : public Interpolation<T>
{
public:
  static OptionSet expected_options();

  BilinearInterpolation(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// The abscissa values of the interpolant
  const Scalar & _X1;
  const Scalar & _X2;

  /// Argument of interpolation
  const Variable<Scalar> & _x1;
  const Variable<Scalar> & _x2;
};
} // namespace neml2
