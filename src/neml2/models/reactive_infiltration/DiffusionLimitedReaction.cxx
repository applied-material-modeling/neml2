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

#include "neml2/models/reactive_infiltration/DiffusionLimitedReaction.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(DiffusionLimitedReaction);

OptionSet
DiffusionLimitedReaction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the void fraction rate of change";

  options.set_input("product_inner_radius") = VariableName{"state", "ri"};
  options.set("product_inner_radius").doc() = "Inner radius of the product phase";
  options.set_input("solid_inner_radius") = VariableName{"state", "ro"};
  options.set("solid_inner_radius").doc() = "Inner raidus of the solid phase";
  options.set<double>("product_dummy_thickness") = 0.01;
  options.set("product_dummy_thickness").doc() = "Minimum product thickness to avoid division by 0";

  options.set_input("liquid_reactivity") = VariableName{"state", "R_l"};
  options.set("liquid_reactivity").doc() = "Reactivity of the liquid phase, between 0 and 1";
  options.set_input("solid_reactivity") = VariableName{"state", "R_s"};
  options.set("solid_reactivity").doc() = "Reactivity of the solid phase, between 0 and 1";

  options.set_output("reaction_rate") = VariableName{"state", "alpha_rate"};
  options.set("reaction_rate").doc() = "Product phase substance (mol/V) rate of change";

  options.set_parameter<TensorName<Scalar>>("diffusion_coefficient");
  options.set("diffusion_coefficient").doc() =
      "Diffusion coefficient of the rate-limiting species in the product phase";

  options.set<double>("molar_volume");
  options.set("molar_volume").doc() = "Molar volume of the rate-limiting (liquid) species";

  return options;
}

DiffusionLimitedReaction::DiffusionLimitedReaction(const OptionSet & options)
  : Model(options),
    _ri(declare_input_variable<Scalar>("product_inner_radius")),
    _ro(declare_input_variable<Scalar>("solid_inner_radius")),
    _delta(options.get<double>("product_dummy_thickness")),
    _R_l(declare_input_variable<Scalar>("liquid_reactivity")),
    _R_s(declare_input_variable<Scalar>("solid_reactivity")),
    _rate(declare_output_variable<Scalar>("reaction_rate")),
    _D(declare_parameter<Scalar>("D", "diffusion_coefficient")),
    _omega(options.get<double>("molar_volume"))
{
}

void
DiffusionLimitedReaction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivatives not implemented");

  const auto factor = 2 * _D * _R_l * _R_s / _omega;
  const auto ratio = (_ro + _ri) / (_ro - _ri + _delta);

  if (out)
  {
    _rate = factor * ratio;
  }

  if (dout_din)
  {
    const auto drate = factor / (_ro - _ri + _delta) / (_ro - _ri + _delta);

    _rate.d(_ri) = drate * (2 * _ro + _delta);
    _rate.d(_ro) = drate * (-2 * _ri + _delta);
    _rate.d(_R_l) = 2 * _D * _R_s / _omega * ratio;
    _rate.d(_R_s) = 2 * _D * _R_l / _omega * ratio;
  }
}
} // namespace neml2
