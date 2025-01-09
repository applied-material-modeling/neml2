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

#include "neml2/tensors/Quaternion.h"

#include "neml2/tensors/R2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/Tensor.h"

#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/bvv.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/acos.h"
#include "neml2/tensors/functions/minimum.h"
#include "neml2/tensors/functions/linalg/vecdot.h"

namespace neml2
{

// TODO: replace at::cat with base_cat
Quaternion::Quaternion(const Rot & r)
  : Quaternion(base_cat(
        {((1 - r.norm_sq()) / (1 + r.norm_sq())).base_unsqueeze(-1), 2 * r / (1 + r.norm_sq())},
        -1))
{
}

Quaternion
Quaternion::fill(const Real & s,
                 const Real & q1,
                 const Real & q2,
                 const Real & q3,
                 const TensorOptions & options)
{
  return Quaternion::fill(
      Scalar(s, options), Scalar(q1, options), Scalar(q2, options), Scalar(q3, options));
}

Quaternion
Quaternion::fill(const Scalar & s, const Scalar & q1, const Scalar & q2, const Scalar & q3)
{
  return Quaternion(base_stack({s, q1, q2, q3}, -1));
}

Scalar
Quaternion::operator()(Size i) const
{
  return PrimitiveTensor<Quaternion, 4>::base_index({i});
}

R2
Quaternion::to_R2() const
{
  const Quaternion & q = *this;

  Scalar v1s = q(1) * q(1);
  Scalar v2s = q(2) * q(2);
  Scalar v3s = q(3) * q(3);

  return R2::fill(1 - 2 * v2s - 2 * v3s,
                  2 * (q(1) * q(2) - q(3) * q(0)),
                  2 * (q(1) * q(3) + q(2) * q(0)),
                  2 * (q(1) * q(2) + q(3) * q(0)),
                  1 - 2 * v1s - 2 * v3s,
                  2 * (q(2) * q(3) - q(1) * q(0)),
                  2 * (q(1) * q(3) - q(2) * q(0)),
                  2 * (q(2) * q(3) + q(1) * q(0)),
                  1 - 2 * v1s - 2 * v2s);
}

Scalar
Quaternion::dot(const Quaternion & other) const
{
  return bvv(*this, other);
}

Scalar
Quaternion::dist(const Quaternion & other) const
{
  Scalar dp = neml2::abs(linalg::vecdot(*this, other));
  return 2.0 * neml2::acos(neml2::minimum(dp, Scalar::ones_like(dp)));
}
} // namespace neml2
