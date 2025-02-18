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

#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/R5.h"
#include "neml2/tensors/SSFR5.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/SSSSR8.h"
#include "neml2/tensors/R8.h"
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/mandel_notation.h"
#include "neml2/tensors/functions/bmv.h"
#include "neml2/tensors/functions/linalg/inv.h"

namespace neml2
{

SSR4::SSR4(const R4 & T)
  : SSR4(full_to_mandel(
        full_to_mandel((T + T.transpose_minor() + R4(T.transpose(0, 1)) + R4(T.transpose(2, 3))) /
                       4.0),
        1))
{
}

SSR4
SSR4::identity(const TensorOptions & options)
{
  return SSR4::create({{1, 1, 1, 0, 0, 0},
                       {1, 1, 1, 0, 0, 0},
                       {1, 1, 1, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0}},
                      options);
}

SSR4
SSR4::identity_C1(const TensorOptions & options)
{
  return SSR4::create({{1, 0, 0, 0, 0, 0},
                       {0, 1, 0, 0, 0, 0},
                       {0, 0, 1, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0}},
                      options);
}

SSR4
SSR4::identity_C2(const TensorOptions & options)
{
  return SSR4::create({{0, 1, 1, 0, 0, 0},
                       {1, 0, 1, 0, 0, 0},
                       {1, 1, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0}},
                      options);
}

SSR4
SSR4::identity_C3(const TensorOptions & options)
{
  return SSR4::create({{0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 1, 0, 0},
                       {0, 0, 0, 0, 1, 0},
                       {0, 0, 0, 0, 0, 1}},
                      options);
}

SSR4
SSR4::identity_sym(const TensorOptions & options)
{
  return SSR4(at::eye(6, options), 0);
}

SSR4
SSR4::identity_vol(const TensorOptions & options)
{
  return SSR4::identity(options) / 3;
}

SSR4
SSR4::identity_dev(const TensorOptions & options)
{
  return SSR4::identity_sym(options) - SSR4::identity(options) / 3;
}

SSR4
SSR4::isotropic_E_nu(const Scalar & E, const Scalar & nu)
{
  neml_assert_broadcastable_dbg(E, nu);

  const auto zero = Scalar::zeros_like(E);
  const auto pf = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const auto C1 = (1.0 - nu) * pf;
  const auto C2 = nu * pf;
  const auto C4 = (1.0 - 2.0 * nu) * pf;

  return SSR4::fill_C1_C2_C3(C1, C2, C4);
}

SSR4
SSR4::isotropic_E_nu(const Real & E, const Real & nu, const TensorOptions & options)
{
  return SSR4::isotropic_E_nu(Scalar(E, options), Scalar(nu, options));
}

SSR4
SSR4::fill_C1_C2_C3(const Scalar & C1, const Scalar & C2, const Scalar & C3)
{
  neml_assert_broadcastable_dbg(C1, C2, C3);

  return C1 * identity_C1(C1.options()) + C2 * identity_C2(C2.options()) +
         C3 * identity_C3(C3.options());
}

SSR4
SSR4::fill_C1_C2_C3(const Real & C1,
                    const Real & C2,
                    const Real & C3,
                    const TensorOptions & options)
{
  return SSR4::fill_C1_C2_C3(Scalar(C1, options), Scalar(C2, options), Scalar(C3, options));
}

SSSSR8
SSR4::identity_map(const TensorOptions & options)
{
  auto I = at::eye(6, options);
  return SSSSR8(at::einsum("ik,jl", {I, I}));
}

SSR4
SSR4::rotate(const Rot & r) const
{
  return R4(*this).rotate(r);
}

SSFR5
SSR4::drotate(const Rot & r) const
{
  auto dR = R4(*this).drotate(r);
  return full_to_mandel(full_to_mandel(dR), 1);
}

SSSSR8
SSR4::drotate_self(const Rot & r) const
{
  auto R = r.euler_rodrigues();
  auto Tsym = 0.25 * (at::einsum("...ma,...nb,...oc,...pd->...mnopabcd", {R, R, R, R}) +
                      at::einsum("...mb,...na,...od,...pc->...mnopabcd", {R, R, R, R}) +
                      at::einsum("...mb,...na,...oc,...pd->...mnopabcd", {R, R, R, R}) +
                      at::einsum("...ma,...nb,...od,...pc->...mnopabcd", {R, R, R, R}));
  return SSSSR8(full_to_mandel(
      full_to_mandel(full_to_mandel(full_to_mandel(R8(Tsym, R.batch_dim()), 0), 1), 2), 3));
}

Scalar
SSR4::operator()(Size i, Size j, Size k, Size l) const
{
  const auto a = mandel_reverse_index[i][j];
  const auto b = mandel_reverse_index[k][l];
  return base_index({a, b}) / (mandel_factor(a) * mandel_factor(b));
}

SSR4
SSR4::inverse() const
{
  return linalg::inv(*this);
}

SSSSR8
SSR4::dinverse() const
{
  auto SI = this->inverse();
  return SSSSR8(-at::einsum("...ik,...lj->...ijkl", {SI, SI}));
}

SSR4
SSR4::transpose_minor() const
{
  return *this;
}

SSR4
SSR4::transpose_major() const
{
  return TensorBase<SSR4>::base_transpose(0, 1);
}

SR2
operator*(const SSR4 & a, const SR2 & b)
{
  return SR2(bmv(a, b));
}

SR2
operator*(const SR2 & a, const SSR4 & b)
{
  return SR2(bmv(b.transpose_major(), a));
}

SSR4
operator*(const SSR4 & a, const SSR4 & b)
{
  return SSR4(at::matmul(a, b));
}
} // namespace neml2
