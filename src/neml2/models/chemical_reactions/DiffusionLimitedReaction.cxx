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

#include "neml2/models/chemical_reactions/DiffusionLimitedReaction.h"

namespace neml2
{
register_NEML2_object(DiffusionLimitedReaction);

OptionSet
DiffusionLimitedReaction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the void fraction rate of change";

  options.add_input("product_inner_radius", "Inner radius of the product phase");
  options.add_input("solid_inner_radius", "Inner radius of the solid phase");
  options.add<double>(
      "product_dummy_thickness", 0.01, "Minimum product thickness to avoid division by 0");

  options.add_input("liquid_reactivity", "Reactivity of the liquid phase, between 0 and 1");
  options.add_input("solid_reactivity", "Reactivity of the solid phase, between 0 and 1");

  options.add_output("reaction_rate", "Product phase substance volumetric rate of change");

  options.add_parameter<Scalar>(
      "diffusion_coefficient",
      "Diffusion coefficient of the rate-limiting species in the product phase");

  options.add<double>("molar_volume", "Molar volume of the rate-limiting (liquid) species");

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
DiffusionLimitedReaction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto factor = 2 * _D * _R_l * _R_s / _omega;
  const auto ratio = _ro / (_ro - _ri + _delta);

  if (out)
  {
    _rate = factor * ratio;
  }

  if (dout_din)
  {
    const auto drate = factor / (_ro - _ri + _delta) / (_ro - _ri + _delta);

    _rate.d(_ri) = drate * _ro;
    _rate.d(_ro) = drate * (-_ri + _delta);
    _rate.d(_R_l) = 2 * _D * _R_s / _omega * ratio;
    _rate.d(_R_s) = 2 * _D * _R_l / _omega * ratio;
  }
}
} // namespace neml2
