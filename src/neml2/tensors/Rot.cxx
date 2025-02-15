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

#include <ATen/Context.h>

#include "neml2/tensors/Rot.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/Quaternion.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/sin.h"
#include "neml2/tensors/functions/cos.h"
#include "neml2/tensors/functions/tan.h"
#include "neml2/tensors/functions/asin.h"
#include "neml2/tensors/functions/acos.h"
#include "neml2/tensors/functions/minimum.h"
#include "neml2/tensors/functions/deg2rad.h"
#include "neml2/tensors/functions/fmod.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/diag_embed.h"

namespace neml2
{
Rot::Rot(const Vec & v)
  : Rot(Tensor(v))
{
}

Rot
Rot::identity(const TensorOptions & options)
{
  return Rot::zeros(options);
}

Rot
Rot::fill_euler_angles(const Vec & v,
                       const std::string & angle_convention,
                       const std::string & angle_type)
{
  auto m = v;

  if (angle_type == "degrees")
    m = neml2::deg2rad(v);
  else
    neml_assert(angle_type == "radians", "Rot angle_type must be either 'degrees' or 'radians'");

  if (angle_convention == "bunge")
  {
    m.base_index_put_({0}, neml2::fmod(m.base_index({0}) - M_PI / 2.0, 2.0 * M_PI));
    m.base_index_put_({1}, neml2::fmod(m.base_index({1}), M_PI));
    m.base_index_put_({2}, neml2::fmod(M_PI / 2.0 - m.base_index({2}), 2.0 * M_PI));
  }
  else if (angle_convention == "roe")
  {
    m.base_index_put_({2}, M_PI - m.base_index({2}));
  }
  else
    neml_assert(angle_convention == "kocks", "Unknown Rot angle_convention " + angle_convention);

  // Make a rotation matrix
  auto M = R2(neml2::base_diag_embed(m));
  auto a = m.base_index({0});
  auto b = m.base_index({1});
  auto c = m.base_index({2});
  M.base_index_put_({0, 0},
                    -neml2::sin(c) * neml2::sin(a) - neml2::cos(c) * neml2::cos(a) * neml2::cos(b));
  M.base_index_put_({0, 1},
                    neml2::sin(c) * neml2::cos(a) - neml2::cos(c) * neml2::sin(a) * neml2::cos(b));
  M.base_index_put_({0, 2}, neml2::cos(c) * neml2::sin(b));
  M.base_index_put_({1, 0},
                    neml2::cos(c) * neml2::sin(a) - neml2::sin(c) * neml2::cos(a) * neml2::cos(b));
  M.base_index_put_({1, 1},
                    -neml2::cos(c) * neml2::cos(a) - neml2::sin(c) * neml2::sin(a) * neml2::cos(b));
  M.base_index_put_({1, 2}, neml2::sin(c) * neml2::sin(b));
  M.base_index_put_({2, 0}, neml2::cos(a) * neml2::sin(b));
  M.base_index_put_({2, 1}, neml2::sin(a) * neml2::sin(b));
  M.base_index_put_({2, 2}, neml2::cos(b));

  // Convert from matrix to vector
  return fill_matrix(M);
}

Rot
Rot::fill_matrix(const R2 & M)
{
  // Get the angle
  auto trace = M(0, 0) + M(1, 1) + M(2, 2);
  auto theta = neml2::acos((trace - 1.0) / 2.0);

  // Get the standard Rod. parameters
  auto scale = neml2::tan(theta / 2.0) / (2.0 * neml2::sin(theta));
  scale.index_put_({theta == 0}, 0.0);
  auto rx = (M(2, 1) - M(1, 2)) * scale;
  auto ry = (M(0, 2) - M(2, 0)) * scale;
  auto rz = (M(1, 0) - M(0, 1)) * scale;

  return fill_rodrigues(rx, ry, rz);
}

Rot
Rot::fill_rodrigues(const Scalar & rx, const Scalar & ry, const Scalar & rz)
{
  // Get the modified Rod. parameters
  auto ns = rx * rx + ry * ry + rz * rz;
  auto f = neml2::sqrt(ns + 1) + 1;

  // Stack and return
  return Rot(base_stack({rx / f, ry / f, rz / f}));
}

Rot
Rot::fill_random(unsigned int n, Size random_seed)
{
  if (random_seed >= 0)
    at::manual_seed(random_seed);
  auto u0 = Scalar(at::rand({n}, default_tensor_options()));
  auto u1 = Scalar(at::rand({n}, default_tensor_options()));
  auto u2 = Scalar(at::rand({n}, default_tensor_options()));

  auto w = neml2::sqrt(1.0 - u0) * neml2::sin(2.0 * M_PI * u1);
  auto x = neml2::sqrt(1.0 - u0) * neml2::cos(2.0 * M_PI * u1);
  auto y = neml2::sqrt(u0) * neml2::sin(2.0 * M_PI * u2);
  auto z = neml2::sqrt(u0) * neml2::cos(2.0 * M_PI * u2);

  auto quats = Quaternion(base_stack({w, x, y, z}));

  return fill_matrix(quats.to_R2());
}

Rot
Rot::inverse() const
{
  return -(*this);
}

R2
Rot::euler_rodrigues() const
{
  auto rr = norm_sq();
  auto E = R3::levi_civita(options());
  auto W = R2::skew(*this);

  return 1.0 / neml2::pow(1 + rr, 2.0) *
         (neml2::pow(1 + rr, 2.0) * R2::identity(options()) + 4 * (1.0 - rr) * W + 8.0 * W * W);
}

R3
Rot::deuler_rodrigues() const
{
  auto rr = norm_sq();
  auto I = R2::identity(options());
  auto E = R3::levi_civita(options());
  auto W = R2::skew(*this);

  return 8.0 * (rr - 3.0) / neml2::pow(1.0 + rr, 3.0) * R3(at::einsum("...ij,...k", {W, *this})) -
         32.0 / neml2::pow(1 + rr, 3.0) * R3(at::einsum("...ij,...k", {(W * W), *this})) -
         4.0 * (1 - rr) / neml2::pow(1.0 + rr, 2.0) * R3(at::einsum("...kij->...ijk", {E})) -
         8.0 / neml2::pow(1.0 + rr, 2.0) *
             R3(at::einsum("...kim,...mj", {E, W}) + at::einsum("...im,...kmj", {W, E}));
}

Rot
Rot::rotate(const Rot & r) const
{
  return r * (*this);
}

R2
Rot::drotate(const Rot & r) const
{
  auto r1 = *this;

  auto rr1 = r1.norm_sq();
  auto rr2 = r.norm_sq();
  auto d = 1.0 + rr1 * rr2 - 2 * Vec(r1).dot(r);
  auto r3 = rotate(r);
  auto I = R2::identity(options());

  return 1.0 / d *
         (-Vec(r3).outer(2 * rr1 * Vec(r) - 2.0 * Vec(r1)) - 2 * Vec(r1).outer(Vec(r)) +
          (1 - rr1) * I - 2 * R2::skew(r1));
}

R2
Rot::drotate_self(const Rot & r) const
{
  auto r2 = *this;

  auto rr1 = r.norm_sq();
  auto rr2 = r2.norm_sq();
  auto d = 1.0 + rr1 * rr2 - 2 * Vec(r).dot(r2);
  auto r3 = rotate(r);
  auto I = R2::identity(options());

  return 1.0 / d *
         (-Vec(r3).outer(2 * rr1 * Vec(r2) - 2.0 * Vec(r)) - 2 * Vec(r).outer(Vec(r2)) +
          (1 - rr1) * I + 2 * R2::skew(r));
}

Rot
Rot::shadow() const
{
  return -*this / norm_sq();
}

R2
Rot::dshadow() const
{
  auto ns = norm_sq();
  return (2.0 / ns * Vec(*this).outer(*this) - R2::identity(options())) / ns;
}

Scalar
Rot::dist(const Rot & r2) const
{
  const auto r1s = this->shadow();
  const auto r2s = r2.shadow();
  const auto d_r1_r2 = this->gdist(r2);
  const auto d_r1_r2s = this->gdist(r2s);
  const auto d_r2s_r2 = r2s.gdist(r2);
  const auto d_r2s_r1s = r2s.gdist(r1s);

  return neml2::minimum(neml2::minimum(neml2::minimum(d_r1_r2, d_r1_r2s), d_r2s_r2), d_r2s_r1s);
}

Scalar
Rot::gdist(const Rot & r) const
{
  return 4.0 *
         neml2::asin((*this - r).norm() / neml2::sqrt((1.0 + norm_sq()) * (1.0 + r.norm_sq())));
}

Scalar
Rot::dV() const
{
  return 8.0 / M_PI * neml2::pow(1.0 + norm_sq(), -3.0);
}

Rot
operator*(const Rot & r1, const Rot & r2)
{
  auto rr1 = r1.norm_sq();
  auto rr2 = r2.norm_sq();

  return Rot((1 - rr2) * Vec(r1) + (1.0 - rr1) * Vec(r2) - 2.0 * Vec(r2).cross(r1)) /
         (1.0 + rr1 * rr2 - 2 * Vec(r1).dot(r2));
}

} // namemspace neml2
