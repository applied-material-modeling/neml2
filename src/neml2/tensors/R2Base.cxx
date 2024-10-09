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

#include "neml2/misc/math.h"
#include "neml2/tensors/R2Base.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/WR2.h"

namespace neml2
{
template <class Derived>
Derived
R2Base<Derived>::fill(const Real & a, const torch::TensorOptions & options)
{
  return R2Base<Derived>::fill(Scalar(a, options));
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Scalar & a)
{
  auto zero = torch::zeros_like(a);
  return Derived(torch::stack({torch::stack({a, zero, zero}, -1),
                               torch::stack({zero, a, zero}, -1),
                               torch::stack({zero, zero, a}, -1)},
                              -2),
                 a.batch_sizes());
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Real & a11,
                      const Real & a22,
                      const Real & a33,
                      const torch::TensorOptions & options)
{
  return R2Base<Derived>::fill(Scalar(a11, options), Scalar(a22, options), Scalar(a33, options));
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Scalar & a11, const Scalar & a22, const Scalar & a33)
{
  auto zero = torch::zeros_like(a11);
  return Derived(torch::stack({torch::stack({a11, zero, zero}, -1),
                               torch::stack({zero, a22, zero}, -1),
                               torch::stack({zero, zero, a33}, -1)},
                              -2),
                 a11.batch_sizes());
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Real & a11,
                      const Real & a22,
                      const Real & a33,
                      const Real & a23,
                      const Real & a13,
                      const Real & a12,
                      const torch::TensorOptions & options)
{
  return R2Base<Derived>::fill(Scalar(a11, options),
                               Scalar(a22, options),
                               Scalar(a33, options),
                               Scalar(a23, options),
                               Scalar(a13, options),
                               Scalar(a12, options));
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Scalar & a11,
                      const Scalar & a22,
                      const Scalar & a33,
                      const Scalar & a23,
                      const Scalar & a13,
                      const Scalar & a12)
{
  return Derived(torch::stack({torch::stack({a11, a12, a13}, -1),
                               torch::stack({a12, a22, a23}, -1),
                               torch::stack({a13, a23, a33}, -1)},
                              -2),
                 a11.batch_sizes());
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Real & a11,
                      const Real & a12,
                      const Real & a13,
                      const Real & a21,
                      const Real & a22,
                      const Real & a23,
                      const Real & a31,
                      const Real & a32,
                      const Real & a33,
                      const torch::TensorOptions & options)
{
  return R2Base<Derived>::fill(Scalar(a11, options),
                               Scalar(a12, options),
                               Scalar(a13, options),
                               Scalar(a21, options),
                               Scalar(a22, options),
                               Scalar(a23, options),
                               Scalar(a31, options),
                               Scalar(a32, options),
                               Scalar(a33, options));
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Scalar & a11,
                      const Scalar & a12,
                      const Scalar & a13,
                      const Scalar & a21,
                      const Scalar & a22,
                      const Scalar & a23,
                      const Scalar & a31,
                      const Scalar & a32,
                      const Scalar & a33)
{
  return Derived(torch::stack({torch::stack({a11, a12, a13}, -1),
                               torch::stack({a21, a22, a23}, -1),
                               torch::stack({a31, a32, a33}, -1)},
                              -2),
                 a11.batch_sizes());
}

template <class Derived>
Derived
R2Base<Derived>::skew(const Vec & v)
{
  const auto z = torch::zeros_like(v(0));
  return Derived(torch::stack({torch::stack({z, -v(2), v(1)}, -1),
                               torch::stack({v(2), z, -v(0)}, -1),
                               torch::stack({-v(1), v(0), z}, -1)},
                              -2),
                 v.batch_sizes());
}

template <class Derived>
Derived
R2Base<Derived>::identity(const torch::TensorOptions & options)
{
  return Derived(torch::eye(3, options), 0);
}

template <class Derived>
Derived
R2Base<Derived>::rotate(const Rot & r) const
{
  return rotate(r.euler_rodrigues());
}

template <class Derived>
Derived
R2Base<Derived>::rotate(const R2 & R) const
{
  return R * R2(*this) * R.transpose();
}

template <class Derived>
R3
R2Base<Derived>::drotate(const Rot & r) const
{
  Derived R = r.euler_rodrigues();
  R3 F = r.deuler_rodrigues();

  return R3(torch::einsum("...itl,...tm,...jm", {F, *this, R}) +
                torch::einsum("...ik,...kt,...jtl", {R, *this, F}),
            broadcast_batch_dim(*this, F, R));
}

template <class Derived>
R4
R2Base<Derived>::drotate(const R2 & R) const
{
  auto I = R2::identity(R.options());
  return torch::einsum("...ik,...jl", {I, R * this->transpose()}) +
         torch::einsum("...jk,...il", {I, R * R2(*this)});
}

template <class Derived>
Scalar
R2Base<Derived>::operator()(Size i, Size j) const
{
  return PrimitiveTensor<Derived, 3, 3>::base_index({i, j});
}

template <class Derived>
Derived
R2Base<Derived>::inverse() const
{
  return torch::Tensor::inverse();
}

template <class Derived>
Derived
R2Base<Derived>::transpose() const
{
  return TensorBase<Derived>::base_transpose(0, 1);
}

template <class Derived1, class Derived2, typename, typename>
Vec
operator*(const Derived1 & A, const Derived2 & b)
{
  neml_assert_batch_broadcastable_dbg(A, b);
  return Vec(torch::einsum("...ik,...k", {A, b}), broadcast_batch_dim(A, b));
}

template <class Derived1, class Derived2, typename, typename>
R2
operator*(const Derived1 & A, const Derived2 & B)
{
  neml_assert_broadcastable_dbg(A, B);
  return R2(torch::einsum("...ik,...kj", {A, B}), broadcast_batch_dim(A, B));
}

// template instantiation

// derived classes
template class R2Base<R2>;

// products
template Vec operator*(const R2 & A, const Vec & b);
template R2 operator*(const R2 & A, const R2 & B);
} // namespace neml2
