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
#include "neml2/tensors/math.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
Rot::Rot(const Vec & v)
  : Rot(Tensor(v))
{
}

Rot
Rot::identity(const torch::TensorOptions & options)
{
  return Rot::zeros(options);
}

Rot
Rot::fill_euler_angles(const torch::Tensor & vals,
                       const std::string & angle_convention,
                       const std::string & angle_type)
{
  neml_assert((torch::numel(vals) % 3) == 0,
              "Rot input values should have length divisable by 3 for input type 'euler_angles'");
  auto ten = vals.reshape({-1, 3});

  if (angle_type == "degrees")
    ten = torch::deg2rad(ten);
  else
    neml_assert(angle_type == "radians", "Rot angle_type must be either 'degrees' or 'radians'");

  if (angle_convention == "bunge")
  {
    ten.index_put_({indexing::Ellipsis, 0},
                   torch::fmod(ten.index({indexing::Ellipsis, 0}) - M_PI / 2.0, 2.0 * M_PI));
    ten.index_put_({indexing::Ellipsis, 1}, torch::fmod(ten.index({indexing::Ellipsis, 1}), M_PI));
    ten.index_put_({indexing::Ellipsis, 2},
                   torch::fmod(M_PI / 2.0 - ten.index({indexing::Ellipsis, 2}), 2.0 * M_PI));
  }
  else if (angle_convention == "roe")
  {
    ten.index_put_({indexing::Ellipsis, 2}, M_PI - ten.index({indexing::Ellipsis, 2}));
  }
  else
    neml_assert(angle_convention == "kocks", "Unknown Rot angle_convention " + angle_convention);

  // Make a rotation matrix...
  auto M = torch::zeros({ten.sizes()[0], 3, 3}, vals.options());
  auto a = ten.index({indexing::Ellipsis, 0});
  auto b = ten.index({indexing::Ellipsis, 1});
  auto c = ten.index({indexing::Ellipsis, 2});

  M.index_put_({indexing::Ellipsis, 0, 0},
               -torch::sin(c) * torch::sin(a) - torch::cos(c) * torch::cos(a) * torch::cos(b));
  M.index_put_({indexing::Ellipsis, 0, 1},
               torch::sin(c) * torch::cos(a) - torch::cos(c) * torch::sin(a) * torch::cos(b));
  M.index_put_({indexing::Ellipsis, 0, 2}, torch::cos(c) * torch::sin(b));
  M.index_put_({indexing::Ellipsis, 1, 0},
               torch::cos(c) * torch::sin(a) - torch::sin(c) * torch::cos(a) * torch::cos(b));
  M.index_put_({indexing::Ellipsis, 1, 1},
               -torch::cos(c) * torch::cos(a) - torch::sin(c) * torch::sin(a) * torch::cos(b));
  M.index_put_({indexing::Ellipsis, 1, 2}, torch::sin(c) * torch::sin(b));
  M.index_put_({indexing::Ellipsis, 2, 0}, torch::cos(a) * torch::sin(b));
  M.index_put_({indexing::Ellipsis, 2, 1}, torch::sin(a) * torch::sin(b));
  M.index_put_({indexing::Ellipsis, 2, 2}, torch::cos(b));

  // Convert from matrix to vector
  return fill_matrix(R2(M, 1));
}

Rot
Rot::fill_matrix(const R2 & M)
{
  // Get the angle
  auto trace = M.index({indexing::Ellipsis, 0, 0}) + M.index({indexing::Ellipsis, 1, 1}) +
               M.index({indexing::Ellipsis, 2, 2});
  auto theta = torch::acos((trace - 1.0) / 2.0);

  // Get the standard Rod. parameters
  auto scale = torch::tan(theta / 2.0) / (2.0 * torch::sin(theta));
  scale.index_put_({theta == 0}, 0.0);
  auto rx = (M.index({indexing::Ellipsis, 2, 1}) - M.index({indexing::Ellipsis, 1, 2})) * scale;
  auto ry = (M.index({indexing::Ellipsis, 0, 2}) - M.index({indexing::Ellipsis, 2, 0})) * scale;
  auto rz = (M.index({indexing::Ellipsis, 1, 0}) - M.index({indexing::Ellipsis, 0, 1})) * scale;

  return fill_rodrigues(rx, ry, rz);
}

Rot
Rot::fill_rodrigues(const Scalar & rx, const Scalar & ry, const Scalar & rz)
{
  // Get the modified Rod. parameters
  auto ns = rx * rx + ry * ry + rz * rz;
  auto f = torch::sqrt(torch::Tensor(ns) + torch::tensor(1.0, ns.dtype())) +
           torch::tensor(1.0, ns.dtype());

  // Stack and return
  return Rot(torch::stack({rx / f, ry / f, rz / f}, 1), 1);
}

Rot
Rot::fill_random(unsigned int n, Size random_seed)
{
  if (random_seed >= 0)
    torch::manual_seed(random_seed);
  auto u0 = torch::rand({n}, default_tensor_options());
  auto u1 = torch::rand({n}, default_tensor_options());
  auto u2 = torch::rand({n}, default_tensor_options());

  auto w = torch::sqrt(1.0 - u0) * torch::sin(2.0 * M_PI * u1);
  auto x = torch::sqrt(1.0 - u0) * torch::cos(2.0 * M_PI * u1);
  auto y = torch::sqrt(u0) * torch::sin(2.0 * M_PI * u2);
  auto z = torch::sqrt(u0) * torch::cos(2.0 * M_PI * u2);

  auto quats = Quaternion(torch::stack({w, x, y, z}, 1), 1);

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

  return 1.0 / math::pow(1 + rr, 2.0) *
         (math::pow(1 + rr, 2.0) * R2::identity(options()) + 4 * (1.0 - rr) * W + 8.0 * W * W);
}

R3
Rot::deuler_rodrigues() const
{
  auto rr = norm_sq();
  auto I = R2::identity(options());
  auto E = R3::levi_civita(options());
  auto W = R2::skew(*this);

  return 8.0 * (rr - 3.0) / math::pow(1.0 + rr, 3.0) * R3(torch::einsum("...ij,...k", {W, *this})) -
         32.0 / math::pow(1 + rr, 3.0) * R3(torch::einsum("...ij,...k", {(W * W), *this})) -
         4.0 * (1 - rr) / math::pow(1.0 + rr, 2.0) * R3(torch::einsum("...kij->...ijk", {E})) -
         8.0 / math::pow(1.0 + rr, 2.0) *
             R3(torch::einsum("...kim,...mj", {E, W}) + torch::einsum("...im,...kmj", {W, E}));
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
  auto d = 1.0 + rr1 * rr2 - 2 * r1.dot(r);
  auto r3 = this->rotate(r);
  auto I = R2::identity(options());

  return 1.0 / d *
         (-r3.outer(2 * rr1 * r - 2.0 * r1) - 2 * r1.outer(r) + (1 - rr1) * I - 2 * R2::skew(r1));
}

R2
Rot::drotate_self(const Rot & r) const
{
  auto r2 = *this;

  auto rr1 = r.norm_sq();
  auto rr2 = r2.norm_sq();
  auto d = 1.0 + rr1 * rr2 - 2 * r.dot(r2);
  auto r3 = this->rotate(r);
  auto I = R2::identity(options());

  return 1.0 / d *
         (-r3.outer(2 * rr1 * r2 - 2.0 * r) - 2 * r.outer(r2) + (1 - rr1) * I + 2 * R2::skew(r));
}

Rot
Rot::shadow() const
{
  return -*this / this->norm_sq();
}

R2
Rot::dshadow() const
{
  auto ns = this->norm_sq();

  return (2.0 / ns * this->outer(*this) - R2::identity(options())) / ns;
}

Scalar
Rot::dist(const Rot & r) const
{
  auto st = this->shadow();
  auto sr = r.shadow();

  return this->gdist(r).minimum(this->gdist(sr)).minimum(sr.gdist(r)).minimum(sr.gdist(st));
}

Scalar
Rot::gdist(const Rot & r) const
{
  return 4.0 * math::arcsin((*this - r).norm() /
                            math::sqrt((1.0 + this->norm_sq()) * (1.0 + r.norm_sq())));
}

Scalar
Rot::dV() const
{
  return 8.0 / M_PI * math::pow(1.0 + this->norm_sq(), -3.0);
}

Rot
operator*(const Rot & r1, const Rot & r2)
{
  auto rr1 = r1.norm_sq();
  auto rr2 = r2.norm_sq();

  return ((1 - rr2) * r1 + (1.0 - rr1) * r2 - 2.0 * r2.cross(r1)) /
         (1.0 + rr1 * rr2 - 2 * r1.dot(r2));
}

} // namemspace neml2
