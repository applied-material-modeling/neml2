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

#include "neml2/tensors/functions/det.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/functions/symmetrization.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
Scalar
det(const Tensor & A)
{
  neml_assert_dbg(A.base_dim() == 2,
                  "Determinant can only be computed for second order tensors. Got base dim ",
                  A.base_dim(),
                  ".");
  return Scalar(at::det(A), A.dynamic_sizes(), A.intmd_dim());
}

Scalar
det(const R2 & A)
{
  const auto comps = at::split(A.base_flatten(), 1, -1);
  const auto & a = comps[0];
  const auto & b = comps[1];
  const auto & c = comps[2];
  const auto & d = comps[3];
  const auto & e = comps[4];
  const auto & f = comps[5];
  const auto & g = comps[6];
  const auto & h = comps[7];
  const auto & i = comps[8];
  const auto det = a * (e * i - h * f) - b * (d * i - g * f) + c * (d * h - e * g);
  return Scalar(det, 0).batch_reshape(A.dynamic_sizes(), A.intmd_sizes());
}

Scalar
det(const SR2 & A)
{
  const auto comps = at::split(A, 1, -1);
  const auto & a = comps[0];
  const auto & e = comps[1];
  const auto & i = comps[2];
  const auto f = comps[3] / mandel_factor(3);
  const auto c = comps[4] / mandel_factor(4);
  const auto b = comps[5] / mandel_factor(5);
  const auto det = a * (e * i - f * f) - b * (b * i - c * f) + c * (b * f - e * c);
  return Scalar(det, 0).batch_reshape(A.dynamic_sizes(), A.intmd_sizes());
}

Scalar
det(const WR2 & A)
{
  return Scalar::zeros(A.options()).batch_expand_as(A);
}
} // namespace neml2
