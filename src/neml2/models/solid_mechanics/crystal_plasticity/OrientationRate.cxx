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

#include "neml2/models/solid_mechanics/crystal_plasticity/OrientationRate.h"

#include "neml2/tensors/WR2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/WWR4.h"
#include "neml2/tensors/WSR4.h"

namespace neml2
{
register_NEML2_object(OrientationRate);

OptionSet
OrientationRate::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() =
      "Defines the rate of the crystal orientations as a spin given by \\f$ \\Omega^e = "
      "w - w^p - \\varepsilon d^p + d^p \\varepsilon \\f$ where \\f$ \\Omega^e = \\dot{Q} Q^T "
      "\\f$, \\f$ Q \\f$ is the orientation, \\f$ w \\f$ is the vorticity, \\f$ w^p \\f$ is the "
      "plastic vorticity, \\f$ d^p \\f$ is the plastic deformation rate, and \\f$ \\varepsilon "
      "\\f$ is the elastic stretch.";

  options.set_output("orientation_rate") = VariableName(STATE, "orientation_rate");
  options.set("orientation_rate").doc() = "The name of the orientation rate (spin)";

  options.set_input("elastic_strain") = VariableName(STATE, "elastic_strain");
  options.set("elastic_strain").doc() = "The name of the elastic strain tensor";

  options.set_input("vorticity") = VariableName(FORCES, "vorticity");
  options.set("vorticity").doc() = "The name of the voriticty tensor";

  options.set_input("plastic_deformation_rate") =
      VariableName(STATE, "internal", "plastic_deformation_rate");
  options.set("plastic_deformation_rate").doc() = "The name of the plastic deformation rate";

  options.set_input("plastic_vorticity") = VariableName(STATE, "internal", "plastic_vorticity");
  options.set("plastic_vorticity").doc() = "The name of the plastic vorticity";
  return options;
}

OrientationRate::OrientationRate(const OptionSet & options)
  : Model(options),
    _R_dot(declare_output_variable<WR2>("orientation_rate")),
    _e(declare_input_variable<SR2>("elastic_strain")),
    _w(declare_input_variable<WR2>("vorticity")),
    _dp(declare_input_variable<SR2>("plastic_deformation_rate")),
    _wp(declare_input_variable<WR2>("plastic_vorticity"))
{
}

static WR2
multiply_and_make_skew(const SR2 & a, const SR2 & b)
{
  auto A = R2(a);
  auto B = R2(b);

  return WR2(A * B - B * A);
}

static WSR4
d_multiply_and_make_skew_d_first(const SR2 & b)
{
  auto I = R2::identity(b.options());
  auto B = R2(b);
  return R4::einsum("...ia,...bj->...ijab", {I, B}) - R4::einsum("...ia,...jb->...ijab", {B, I});
}

static WSR4
d_multiply_and_make_skew_d_second(const SR2 & a)
{
  auto I = R2::identity(a.options());
  auto A = R2(a);
  return R4::einsum("...ia,...jb->...ijab", {A, I}) - R4::einsum("...ia,...bj->...ijab", {I, A});
}

void
OrientationRate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _R_dot = _w - _wp + multiply_and_make_skew(_dp(), _e());

  if (dout_din)
  {
    const auto I = WWR4::identity(_w.options());

    if (_e.is_dependent())
      _R_dot.d(_e) = d_multiply_and_make_skew_d_second(_dp());

    if (_w.is_dependent())
      _R_dot.d(_w) = I;

    if (_dp.is_dependent())
      _R_dot.d(_dp) = d_multiply_and_make_skew_d_first(_e());

    if (_wp.is_dependent())
      _R_dot.d(_wp) = -I;
  }
}
} // namespace neml2
