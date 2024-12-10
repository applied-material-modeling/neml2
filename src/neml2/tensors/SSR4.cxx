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
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/R5.h"
#include "neml2/tensors/SSFR5.h"
#include "neml2/tensors/Rot.h"

namespace neml2
{
SSR4::SSR4(const R4 & T)
  : SSR4(math::full_to_mandel(
        math::full_to_mandel(
            (T + T.transpose_minor() + R4(T.transpose(0, 1)) + R4(T.transpose(2, 3))) / 4.0),
        1))
{
}

SSR4
SSR4::identity(const torch::TensorOptions & options)
{
  return SSR4(torch::tensor({{1, 1, 1, 0, 0, 0},
                             {1, 1, 1, 0, 0, 0},
                             {1, 1, 1, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0},
                             {0, 0, 0, 0, 0, 0}},
                            options),
              0);
}

SSR4
SSR4::identity_sym(const torch::TensorOptions & options)
{
  return SSR4(torch::eye(6, options), 0);
}

SSR4
SSR4::identity_vol(const torch::TensorOptions & options)
{
  return SSR4::identity(options) / 3;
}

SSR4
SSR4::identity_dev(const torch::TensorOptions & options)
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

  return SSR4(torch::stack({torch::stack({C1, C2, C2, zero, zero, zero}, -1),
                            torch::stack({C2, C1, C2, zero, zero, zero}, -1),
                            torch::stack({C2, C2, C1, zero, zero, zero}, -1),
                            torch::stack({zero, zero, zero, C4, zero, zero}, -1),
                            torch::stack({zero, zero, zero, zero, C4, zero}, -1),
                            torch::stack({zero, zero, zero, zero, zero, C4}, -1)},
                           -1),
              E.batch_sizes());
}

SSR4
SSR4::isotropic_E_nu(const Real & E, const Real & nu, const torch::TensorOptions & options)
{
  return SSR4::isotropic_E_nu(Scalar(E, options), Scalar(nu, options));
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
  return math::full_to_mandel(math::full_to_mandel(dR), 1);
}

Scalar
SSR4::operator()(Size i, Size j, Size k, Size l) const
{
  const auto a = math::mandel_reverse_index[i][j];
  const auto b = math::mandel_reverse_index[k][l];
  return base_index({a, b}) / (math::mandel_factor(a) * math::mandel_factor(b));
}

SSR4
SSR4::inverse() const
{
  return math::linalg::inv(*this);
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
  neml_assert_batch_broadcastable_dbg(a, b);
  return SR2(torch::matmul(a, b.unsqueeze(-1)).squeeze(-1), broadcast_batch_dim(a, b));
}

SR2
operator*(const SR2 & a, const SSR4 & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  return SR2(torch::matmul(a.unsqueeze(-2), b).squeeze(-2), broadcast_batch_dim(a, b));
}

SSR4
operator*(const SSR4 & a, const SSR4 & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return SSR4(torch::matmul(a, b), broadcast_batch_dim(a, b));
}
} // namespace neml2
