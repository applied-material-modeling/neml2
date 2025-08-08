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

#include "neml2/tensors/functions/linalg/dsptrf.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

namespace neml2::linalg
{

SSR4
dsptrf(const Vec & evals, const R2 & evecs, const Vec & transformed, const Vec & dtransformed)
{
  auto v1 = evecs.col(0);
  auto v2 = evecs.col(1);
  auto v3 = evecs.col(2);

  auto M_11 = v1.self_outer();
  auto M_22 = v2.self_outer();
  auto M_33 = v3.self_outer();
  auto M_12 = SR2(v1.outer(v2));
  auto M_21 = SR2(v2.outer(v1));
  auto M_13 = SR2(v1.outer(v3));
  auto M_31 = SR2(v3.outer(v1));
  auto M_23 = SR2(v2.outer(v3));
  auto M_32 = SR2(v3.outer(v2));

  auto projection_tensor_1 = (Scalar(dtransformed.base_index({0})) * M_11.outer(M_11) +
                              Scalar(dtransformed.base_index({1})) * M_22.outer(M_22) +
                              Scalar(dtransformed.base_index({2})) * M_33.outer(M_33));

  auto theta_12 =
      where(Scalar(evals.base_index({0})) != Scalar(evals.base_index({1})),
            Scalar(0.5 * (transformed.base_index({0}) - transformed.base_index({1})) /
                   (evals.base_index({0}) - evals.base_index({1}))),
            Scalar(0.25 * (dtransformed.base_index({0}) + dtransformed.base_index({1}))));
  auto theta_23 =
      where(Scalar(evals.base_index({1})) != Scalar(evals.base_index({2})),
            Scalar(0.5 * (transformed.base_index({1}) - transformed.base_index({2})) /
                   (evals.base_index({1}) - evals.base_index({2}))),
            Scalar(0.25 * (dtransformed.base_index({1}) + dtransformed.base_index({2}))));
  auto theta_13 =
      where(evals.base_index({0}) != evals.base_index({2}),
            Scalar(0.5 * (transformed.base_index({0}) - transformed.base_index({2})) /
                   (evals.base_index({0}) - evals.base_index({2}))),
            Scalar(0.25 * (dtransformed.base_index({0}) + dtransformed.base_index({2}))));

  auto projection_tensor_2 =
      Scalar(theta_12) *
          (M_12.outer(M_12) + M_12.outer(M_21) + M_21.outer(M_21) + M_21.outer(M_12)) +
      Scalar(theta_13) *
          (M_13.outer(M_13) + M_13.outer(M_31) + M_31.outer(M_31) + M_31.outer(M_13)) +
      Scalar(theta_23) *
          (M_23.outer(M_23) + M_23.outer(M_32) + M_32.outer(M_32) + M_32.outer(M_23));

  return SSR4(projection_tensor_1 + projection_tensor_2);
}
} // namespace neml2::linalg
