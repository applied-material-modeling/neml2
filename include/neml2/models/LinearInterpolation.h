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
  static T2 mask(const T2 & in, const Scalar & m);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// The abscissa values of the interpolant
  const Scalar & _X;

  /// Argument of interpolation
  const Variable<Scalar> & _x;
};

template <typename T>
template <typename T2>
T2
LinearInterpolation<T>::mask(const T2 & in, const Scalar & m)
{
  /// extracting the bool tensor
  at::Tensor mt = m.data();

  /// keeping only the batch shape safe not base
  const auto B = m.batch_sizes().slice(0, -1);

  /// making sure if bool
  if (mt.dtype() != at::kBool)
    mt = mt.to(at::kBool);

  /// getting the index of true only. If all false argmax falls back with 0 i.e. first index
  at::Tensor idx = mt.to(at::kLong).argmax(-1);

  /// getting the input tensor data expanded to match mask's batch structure
  at::Tensor in_data = in.batch_expand_as(m).data();

  /// getting interpolation dim
  auto in_shape = in_data.sizes();
  auto idx_shape = idx.sizes();
  int interp_dim = static_cast<int>(idx_shape.size());

  /// adding dim to match input tensor dimensions manually (creating shape)
  at::Tensor idx_exp = idx;
  for (int i = interp_dim; i < static_cast<int>(in_shape.size()); i++)
  {
    if (i == interp_dim)
      idx_exp = idx_exp.unsqueeze(i);
    else
      idx_exp = idx_exp.unsqueeze(-1);
  }
  /// expanding to match input shape
  std::vector<int64_t> expand_shape(in_shape.begin(), in_shape.end());
  expand_shape[interp_dim] = 1;
  /// but keep interpolation dim as size 1
  idx_exp = idx_exp.expand(expand_shape);

  at::Tensor gathered = at::take_along_dim(in_data, idx_exp, interp_dim);
  gathered = gathered.squeeze(interp_dim);

  /// wraping back into T2 and restoring batch shape B
  return T2(gathered).batch_reshape(B);
}

} // namespace neml2
