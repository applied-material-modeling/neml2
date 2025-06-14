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

#include "neml2/tensors/SR2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/SFR3.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/SFFR4.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/SWR4.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/assertions.h"

#include "neml2/tensors/mandel_notation.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/sum.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/linalg/vecdot.h"

namespace neml2
{

SR2::SR2(const R2 & T)
  : SR2(full_to_mandel((T + T.transpose()) / 2.0))
{
}

SR2
SR2::fill(const CScalar & a, const TensorOptions & options)
{
  return SR2::fill(Scalar(a, options));
}

SR2
SR2::fill(const Scalar & a)
{
  auto zero = Scalar::zeros_like(a);
  return SR2(base_stack({a, a, a, zero, zero, zero}, -1));
}

SR2
SR2::fill(const CScalar & a11,
          const CScalar & a22,
          const CScalar & a33,
          const TensorOptions & options)
{
  return SR2::fill(Scalar(a11, options), Scalar(a22, options), Scalar(a33, options));
}

SR2
SR2::fill(const Scalar & a11, const Scalar & a22, const Scalar & a33)
{
  auto zero = Scalar::zeros_like(a11);
  return SR2(base_stack({a11, a22, a33, zero, zero, zero}, -1));
}

SR2
SR2::fill(const CScalar & a11,
          const CScalar & a22,
          const CScalar & a33,
          const CScalar & a23,
          const CScalar & a13,
          const CScalar & a12,
          const TensorOptions & options)
{
  return SR2::fill(Scalar(a11, options),
                   Scalar(a22, options),
                   Scalar(a33, options),
                   Scalar(a23, options),
                   Scalar(a13, options),
                   Scalar(a12, options));
}

SR2
SR2::fill(const Scalar & a11,
          const Scalar & a22,
          const Scalar & a33,
          const Scalar & a23,
          const Scalar & a13,
          const Scalar & a12)
{
  return SR2(base_stack(
      {a11, a22, a33, mandel_factor(3) * a23, mandel_factor(4) * a13, mandel_factor(5) * a12}, -1));
}

SR2
SR2::identity(const TensorOptions & options)
{
  return SR2::create({1, 1, 1, 0, 0, 0}, options);
}

SSR4
SR2::identity_map(const TensorOptions & options)
{
  return SSR4::identity_sym(options);
}

SR2
SR2::rotate(const Rot & r) const
{
  return R2(*this).rotate(r);
}

SR2
SR2::rotate(const R2 & R) const
{
  return R2(*this).rotate(R);
}

SFR3
SR2::drotate(const Rot & r) const
{
  auto dR = R2(*this).drotate(r);
  return full_to_mandel(dR);
}

SFFR4
SR2::drotate(const R2 & R) const
{
  auto dR = R2(*this).drotate(R);
  return full_to_mandel(dR);
}

Scalar
SR2::operator()(Size i, Size j) const
{
  Size a = mandel_reverse_index[i][j];
  return base_index({a}) / mandel_factor(a);
}

Scalar
SR2::tr() const
{
  return Scalar(base_sum(base_index({indexing::Slice(0, 3)}), -1), batch_sizes());
}

SR2
SR2::vol() const
{
  return SR2::fill(tr()) / 3;
}

SR2
SR2::dev() const
{
  return *this - vol();
}

Scalar
SR2::inner(const SR2 & other) const
{
  return linalg::vecdot(*this, other);
}

Scalar
SR2::norm_sq() const
{
  return inner(*this);
}

Scalar
SR2::norm(const CScalar & eps) const
{
  return neml2::sqrt(norm_sq() + eps);
}

SSR4
SR2::outer(const SR2 & other) const
{
  neml_assert_broadcastable_dbg(*this, other);
  return SSR4(at::einsum("...i,...j", {*this, other}), utils::broadcast_batch_dim(*this, other));
}

Scalar
SR2::det() const
{
  const auto comps = at::split(*this, 1, -1);
  const auto & a = comps[0];
  const auto & e = comps[1];
  const auto & i = comps[2];
  const auto f = comps[3] / mandel_factor(3);
  const auto c = comps[4] / mandel_factor(4);
  const auto b = comps[5] / mandel_factor(5);
  const auto det = a * (e * i - f * f) - b * (b * i - c * f) + c * (b * f - e * c);
  return Scalar(det.reshape(batch_sizes().concrete()), batch_sizes());
}

SR2
SR2::inverse() const
{
  const auto comps = at::split(*this, 1, -1);
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
  return SR2(inv, batch_sizes());
}

SR2
SR2::transpose() const
{
  return SR2(*this, batch_sizes());
}

} // namespace neml2
