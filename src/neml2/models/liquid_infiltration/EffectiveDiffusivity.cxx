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

#include "neml2/models/liquid_infiltration/EffectiveDiffusivity.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(EffectiveDiffusivity);
OptionSet
EffectiveDiffusivity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the effective diffusion coeficient from the liquid's condition";

  options.set_parameter<CrossRef<Scalar>>("macroscopic_diffusion_coefficient");
  options.set("macroscopic_diffusion_coefficient").doc() =
      "Macroscopic Diffusion Coefficient to scale from";

  options.set_parameter<CrossRef<Scalar>>("coefficient_liquid_limit");
  options.set("coefficient_liquid_limit").doc() =
      "The coefficient corresponding to the scaling from the liquid's limit";

  options.set_input("scaling_liquid_limit") = VariableName("state", "scaling_liquid_limit");
  options.set("scaling_liquid_limit").doc() =
      "The scaling to the diffusivity that based on the liquid's limit";

  options.set<bool>("const_liquid_limit") = false;
  options.set("const_liquid_limit").doc() =
      "Whether the scaling liquid's limit will be a constant value.";

  options.set_input("switching_liquid_limit") = VariableName("state", "switching_liquid_limit");
  options.set("switching_liquid_limit").doc() =
      "The liquid's switching function to determine when the new coefficient is active";

  options.set_output("effective_diffusion_coefficient") =
      VariableName("state", "effective_diffusion_coefficient");
  options.set("effective_diffusion_coefficient").doc() = "The effective diffusion coefficient.";
  return options;
}

EffectiveDiffusivity::EffectiveDiffusivity(const OptionSet & options)
  : Model(options),
    _D(declare_parameter<Scalar>("Dm", "macroscopic_diffusion_coefficient")),
    _coeff(declare_parameter<Scalar>("cm", "coefficient_liquid_limit")),
    _scale(declare_input_variable<Scalar>("scaling_liquid_limit")),
    _switch(declare_input_variable<Scalar>("switching_liquid_limit")),
    _cond(options.get<bool>("const_liquid_limit")),
    _Deff(declare_output_variable<Scalar>("effective_diffusion_coefficient"))
{
}

void
EffectiveDiffusivity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    if (_cond)
    {
      _Deff = (1.0 + (_coeff - 1.0) * _switch) * _D;
    }
    else
      _Deff = (1.0 + (_coeff * _scale - 1.0) * _switch) * _D;
  }

  if (dout_din)
  {
    if (_cond)
    {
      _Deff.d(_switch) = (_coeff - 1.0) * _D;
      _Deff.d(_scale) = neml2::Scalar::full(0.0);
    }
    else
    {
      _Deff.d(_scale) = _coeff * _switch * _D;
      _Deff.d(_switch) = (_coeff * _scale - 1.0) * _D;
    }
    if (currently_solving_nonlinear_system())
      return;
  }
}
}