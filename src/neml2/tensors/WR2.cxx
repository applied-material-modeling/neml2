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

#include "neml2/tensors/WR2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/WSR4.h"
#include "neml2/tensors/R4.h"

#include "neml2/tensors/mandel_notation.h"
#include "neml2/tensors/functions/tan.h"
#include "neml2/tensors/functions/cos.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
WR2::WR2(const R2 & T)
  : WR2(full_to_skew((T - T.transpose()) / 2.0))
{
}

Scalar
WR2::operator()(Size i, Size j) const
{
  Size a = skew_reverse_index[i][j];
  return base_index({a}) * skew_factor[i][j];
}

Rot
WR2::exp() const
{
  // There are singularities at norm() = 0 and 2*pi
  // To the third order near zero this reduces to
  // *this * (1/4 + 5 * norm^4 / 96)
  // We use this formula for small rotations

  // The other singularity is essentially unavoidable

  // This is what determines which region to sit in
  auto norm2 = norm_sq();

  // So we want the result to be as accurate as machine precision
  auto thresh = std::pow(eps, 1.0 / 3.0);

  // Taylor series
  auto res_taylor = Rot(*this) * (0.25 + 5.0 * norm2 * norm2 / 96.0);

  // Actual definition
  auto res_actual = Rot(*this) * neml2::tan(norm2 / 2.0) / (2.0 * norm2 * neml2::cos(norm2 / 2));

  return Rot(at::where((norm2 > thresh).unsqueeze(-1), res_actual, res_taylor));
}

R2
WR2::dexp() const
{
  // Same singularities as WR2::exp()
  auto norm2 = norm_sq();
  auto thresh = std::pow(eps, 1.0 / 3.0);

  auto res_taylor = 5.0 * norm2 / 24.0 * Vec(*this).outer(Vec(*this)) +
                    (0.25 + 5.0 * norm2 * norm2 / 96.0) * R2::identity(options());

  auto f1 = neml2::tan(norm2 / 2.0) / (2.0 * norm2 * neml2::cos(norm2 / 2));
  auto f2 = (norm2 * neml2::pow(1.0 / neml2::cos(norm2 / 2), 3.0) +
             neml2::tan(norm2 / 2.0) * (norm2 * neml2::tan(norm2 / 2.0) - 2.0) *
                 (1.0 / neml2::cos(norm2 / 2.0))) /
            (2 * norm2 * norm2);

  auto res_actual = f1 * R2::identity(options()) + f2 * Vec(*this).outer(Vec(*this));

  return R2(at::where((norm2 > thresh).unsqueeze(-1).unsqueeze(-1), res_actual, res_taylor));
}

} // namespace neml2
