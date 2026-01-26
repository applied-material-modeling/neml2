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

#include "neml2/tensors/R2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/mm.h"
#include "neml2/tensors/functions/mv.h"
#include "neml2/tensors/functions/vdot.h"
#include "neml2/tensors/functions/symmetrization.h"

namespace neml2
{
R2::R2(const SR2 & S)
  : R2(mandel_to_full(S))
{
}

R2::R2(const WR2 & W)
  : R2(skew_to_full(W))
{
}

R2
R2::fill(const CScalar & a, const TensorOptions & options)
{
  return R2::fill(Scalar(a, options));
}

R2
R2::fill(const Scalar & a)
{
  return R2::fill(a, a, a);
}

R2
R2::fill(const CScalar & a11,
         const CScalar & a22,
         const CScalar & a33,
         const TensorOptions & options)
{
  return R2::fill(Scalar(a11, options), Scalar(a22, options), Scalar(a33, options));
}

R2
R2::fill(const Scalar & a11, const Scalar & a22, const Scalar & a33)
{
  auto zero = Scalar::zeros_like(a11);
  return R2::fill(a11, a22, a33, zero, zero, zero);
}

R2
R2::fill(const CScalar & a11,
         const CScalar & a22,
         const CScalar & a33,
         const CScalar & a23,
         const CScalar & a13,
         const CScalar & a12,
         const TensorOptions & options)
{
  return R2::fill(Scalar(a11, options),
                  Scalar(a22, options),
                  Scalar(a33, options),
                  Scalar(a23, options),
                  Scalar(a13, options),
                  Scalar(a12, options));
}

R2
R2::fill(const Scalar & a11,
         const Scalar & a22,
         const Scalar & a33,
         const Scalar & a23,
         const Scalar & a13,
         const Scalar & a12)
{
  return base_stack({base_stack({a11, a12, a13}, -1),
                     base_stack({a12, a22, a23}, -1),
                     base_stack({a13, a23, a33}, -1)},
                    -2);
}

R2
R2::skew(const Vec & v)
{
  const auto z = Scalar::zeros_like(v(0));
  return base_stack({base_stack({z, -v(2), v(1)}, -1),
                     base_stack({v(2), z, -v(0)}, -1),
                     base_stack({-v(1), v(0), z}, -1)},
                    -2);
}

R2
R2::identity(const TensorOptions & options)
{
  return neml2::Tensor::identity(3, options);
}

R2
R2::rotate(const Rot & r) const
{
  return rotate(r.euler_rodrigues());
}

R2
R2::rotate(const R2 & R) const
{
  return R * R2(*this) * R.transpose();
}

R3
R2::drotate(const Rot & r) const
{
  auto R = r.euler_rodrigues();
  auto F = r.deuler_rodrigues();
  return R3::einsum("...itl,...tm,...jm", {F, *this, R}) +
         R3::einsum("...ik,...kt,...jtl", {R, *this, F});
}

R4
R2::drotate(const R2 & R) const
{
  auto I = R2::identity(R.options());
  return R4::einsum("...ik,...jl", {I, R * transpose()}) +
         R4::einsum("...jk,...il", {I, R * *this});
}

Vec
R2::row(Size i) const
{
  return Vec(base_index({i, indexing::Slice()}), dynamic_sizes(), intmd_dim());
}

Vec
R2::col(Size i) const
{
  return Vec(base_index({indexing::Slice(), i}), dynamic_sizes(), intmd_dim());
}

R2
R2::transpose() const
{
  return TensorBase<R2>::base_transpose(0, 1);
}

R2
operator*(const R2 & A, const R2 & B)
{
  return neml2::mm(A, B);
}

Vec
operator*(const R2 & A, const Vec & b)
{
  return neml2::mv(A, b);
}

} // namespace neml2
