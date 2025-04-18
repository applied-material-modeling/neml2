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

#include "neml2/tensors/R2Base.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/functions/stack.h"

namespace neml2
{
template <class Derived>
Derived
R2Base<Derived>::fill(const Real & a, const TensorOptions & options)
{
  return R2Base<Derived>::fill(Scalar(a, options));
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Scalar & a)
{
  auto zero = Scalar::zeros_like(a);
  return Derived(base_stack({base_stack({a, zero, zero}, -1),
                             base_stack({zero, a, zero}, -1),
                             base_stack({zero, zero, a}, -1)},
                            -2));
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Real & a11,
                      const Real & a22,
                      const Real & a33,
                      const TensorOptions & options)
{
  return R2Base<Derived>::fill(Scalar(a11, options), Scalar(a22, options), Scalar(a33, options));
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Scalar & a11, const Scalar & a22, const Scalar & a33)
{
  auto zero = Scalar::zeros_like(a11);
  return Derived(base_stack({base_stack({a11, zero, zero}, -1),
                             base_stack({zero, a22, zero}, -1),
                             base_stack({zero, zero, a33}, -1)},
                            -2));
}

template <class Derived>
Derived
R2Base<Derived>::fill(const Real & a11,
                      const Real & a22,
                      const Real & a33,
                      const Real & a23,
                      const Real & a13,
                      const Real & a12,
                      const TensorOptions & options)
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
  return Derived(base_stack({base_stack({a11, a12, a13}, -1),
                             base_stack({a12, a22, a23}, -1),
                             base_stack({a13, a23, a33}, -1)},
                            -2));
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
                      const TensorOptions & options)
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
  return Derived(base_stack({base_stack({a11, a12, a13}, -1),
                             base_stack({a21, a22, a23}, -1),
                             base_stack({a31, a32, a33}, -1)},
                            -2));
}

template <class Derived>
Derived
R2Base<Derived>::skew(const Vec & v)
{
  const auto z = Scalar::zeros_like(v(0));
  return Derived(base_stack({base_stack({z, -v(2), v(1)}, -1),
                             base_stack({v(2), z, -v(0)}, -1),
                             base_stack({-v(1), v(0), z}, -1)},
                            -2));
}

template <class Derived>
Derived
R2Base<Derived>::identity(const TensorOptions & options)
{
  return Derived(at::eye(3, options), 0);
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
  auto R = r.euler_rodrigues();
  auto F = r.deuler_rodrigues();

  return R3(at::einsum("...itl,...tm,...jm", {F, *this, R}) +
            at::einsum("...ik,...kt,...jtl", {R, *this, F}));
}

template <class Derived>
R4
R2Base<Derived>::drotate(const R2 & R) const
{
  auto I = R2::identity(R.options());
  return R4(at::einsum("...ik,...jl", {I, R * this->transpose()}) +
            at::einsum("...jk,...il", {I, R * R2(*this)}));
}

template <class Derived>
Scalar
R2Base<Derived>::operator()(Size i, Size j) const
{
  return PrimitiveTensor<Derived, 3, 3>::base_index({i, j});
}

template <class Derived>
Scalar
R2Base<Derived>::det() const
{
  const auto comps = at::split(this->base_flatten(), 1, -1);
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
  return Scalar(det.reshape(this->batch_sizes().concrete()), this->batch_sizes());
}

template <class Derived>
Derived
R2Base<Derived>::inverse() const
{
  const auto comps = at::split(this->base_flatten(), 1, -1);
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
  return Derived(inv, this->batch_sizes());
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
  return Vec(at::einsum("...ik,...k", {A, b}));
}

template <class Derived1, class Derived2, typename, typename>
R2
operator*(const Derived1 & A, const Derived2 & B)
{
  neml_assert_broadcastable_dbg(A, B);
  return R2(at::einsum("...ik,...kj", {A, B}));
}

// template instantiation

// derived classes
template class R2Base<R2>;

// products
template Vec operator*(const R2 & A, const Vec & b);
template R2 operator*(const R2 & A, const R2 & B);
} // namespace neml2
