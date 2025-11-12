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
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/symmetrization.h"
#include "neml2/tensors/functions/inv.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
Tensor
inv(const Tensor & A)
{
  neml_assert_dbg(A.scalar_type() == neml2::kFloat32 || A.scalar_type() == neml2::kFloat64,
                  "Inverse only supports float32 and float64, got",
                  A.scalar_type());
  neml_assert_dbg(
      A.base_dim() == 2 || A.base_dim() == 4,
      "Inverse can only be computed for second order and fourth order tensors. Got base dim ",
      A.base_dim(),
      ".");

  if (A.base_dim() == 2)
    return Tensor(at::inverse(A), A.dynamic_sizes(), A.intmd_dim());

  // A.base_dim() == 4
  neml_assert_dbg(
      A.base_size(0) == A.base_size(1) && A.base_size(2) == A.base_size(3) &&
          A.base_size(0) == A.base_size(2),
      "Inverse of fourth order tensors only implemented for square tensors. Got base shape ",
      A.base_sizes());
  auto n = A.base_size(0);
  auto Ainv = Tensor(at::inverse(A.base_reshape({n * n, n * n})), A.dynamic_sizes(), A.intmd_dim());
  return Ainv.base_reshape(A.base_sizes());
}

Rot
inv(const Rot & R)
{
  return -R;
}

R2
inv(const R2 & A)
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
  const auto cof00 = e * i - h * f;
  const auto cof01 = -(d * i - g * f);
  const auto cof02 = d * h - g * e;
  const auto cof10 = -(b * i - h * c);
  const auto cof11 = a * i - g * c;
  const auto cof12 = -(a * h - g * b);
  const auto cof20 = b * f - e * c;
  const auto cof21 = -(a * f - d * c);
  const auto cof22 = a * e - d * b;
  const auto coft0 = at::cat({cof00, cof10, cof20}, -1);
  const auto coft1 = at::cat({cof01, cof11, cof21}, -1);
  const auto coft2 = at::cat({cof02, cof12, cof22}, -1);
  const auto coft = at::stack({coft0, coft1, coft2}, -2);
  const auto inv = coft / det.unsqueeze(-1);
  return R2(inv, A.dynamic_sizes(), A.intmd_dim());
}

SR2
inv(const SR2 & A)
{
  const auto comps = at::split(A, 1, -1);
  const auto & a = comps[0];
  const auto & e = comps[1];
  const auto & i = comps[2];
  const auto f = comps[3] / mandel_factor(3);
  const auto c = comps[4] / mandel_factor(4);
  const auto b = comps[5] / mandel_factor(5);
  const auto det = a * (e * i - f * f) - b * (b * i - c * f) + c * (b * f - e * c);
  const auto cof00 = e * i - f * f;
  const auto cof01 = -(b * i - c * f);
  const auto cof02 = b * f - c * e;
  const auto cof11 = a * i - c * c;
  const auto cof12 = -(a * f - c * b);
  const auto cof22 = a * e - b * b;
  const auto cof = at::cat({cof00,
                            cof11,
                            cof22,
                            mandel_factor(3) * cof12,
                            mandel_factor(4) * cof02,
                            mandel_factor(5) * cof01},
                           -1);
  const auto inv = cof / det;
  return SR2(inv, A.dynamic_sizes(), A.intmd_dim());
}

R4
inv(const R4 & A)
{
  auto Ainv = Tensor(at::inverse(A.base_reshape({3, 3})), A.dynamic_sizes(), A.intmd_dim());
  return Ainv.base_reshape({3, 3, 3, 3});
}

SSR4
inv(const SSR4 & A)
{
  return SSR4(at::inverse(A), A.dynamic_sizes(), A.intmd_dim());
}
} // namespace neml2
