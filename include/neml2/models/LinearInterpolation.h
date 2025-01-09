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
 * @brief Linearly interpolate the parameter along a single axis.
 *
 * Currently, this object is hard-coded to always interpolate along the last batch dimension.
 * A few examples of tensor shapes are listed below to demonstrate how broadcasting is handled:
 *
 * Example 1: unbatched abscissa, unbatched ordinate (of type R2), unbatched input argument,
 * interpolant size 100
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * abscissa shape: (100;     )
 * ordinate shape: (100; 3, 3)
 *    input shape: (   ;     )
 *   output shape: (   ; 3, 3)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Example 2: unbatched abscissa, unbatched ordinate (of type R2), batched input argument (with
 * batch shape `(2, 3)`), interpolant size 100
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * abscissa shape: (     100;     )
 * ordinate shape: (     100; 3, 3)
 *    input shape: (2, 3    ;     )
 *   output shape: (2, 3    ; 3, 3)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Example 3: unbatched abscissa, batched ordinate (of type R2 and with batch shape `(5, 1)`),
 * batched input argument (with batch shape `(2, 5, 2)`), interpolant size 100
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * abscissa shape: (         100;     )
 * ordinate shape: (   5, 1, 100; 3, 3)
 *    input shape: (2, 5, 2     ;     )
 *   output shape: (2, 5, 2     ; 3, 3)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Example 4: batched abscissa (with batch shape `(7, 8, 1)`), unbatched ordinate (of type R2),
 * batched input argument (with batch shape `(7, 8, 5)`), interpolant size 100
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * abscissa shape: (7, 8, 1, 100;     )
 * ordinate shape: (         100; 3, 3)
 *    input shape: (7, 8, 5     ;     )
 *   output shape: (7, 8, 5     ; 3, 3)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 */
template <typename T>
class LinearInterpolation : public Interpolation<T>
{
public:
  static OptionSet expected_options();

  LinearInterpolation(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

private:
  /**
   * @brief Apply the mask tensor \p m on the input \p in.
   *
   * This method additionally handles the necessary expanding and reshaping based on two
   * assumptions:
   * 1. The mask only selects 1 single batch along the interpolated dimension.
   * 2. We are always interpolating along the last batch dimension.
   *
   * So if some day we relax the 2nd assumption, this method need to be adapted accordingly.
   */
  template <typename T2>
  T2 mask(const T2 & in, const Scalar & m) const;
};

template <typename T>
template <typename T2>
T2
LinearInterpolation<T>::mask(const T2 & in, const Scalar & m) const
{
  // Resulting batch shape
  const auto B = m.batch_sizes().slice(0, -1);
  // Use advanced (boolean) indexing to select the correct interval
  return T2(in.batch_expand_as(m).index({m})).batch_reshape(B);
}
} // namespace neml2
