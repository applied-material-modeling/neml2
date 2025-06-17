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

#include "neml2/models/porous_flow/EffectiveSaturation.h"

namespace neml2
{
register_NEML2_object(EffectiveSaturation);
OptionSet
EffectiveSaturation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the effective saturation, taking the form of \\f$ S = "
      "\\frac{\\frac{\\phi}{\\phi_\\mathrm{max}} - S_r}{1-S_r} \\f$ where \\f$ \\phi \\f$ is the "
      "volume fraction of the flowing fluid, \\f$ \\phi_\\mathrm{max} \\f$ is the maximum "
      "allowable volume fraction and \\f$ S_r \\f$ is the residual saturation.";

  options.set_parameter<TensorName<Scalar>>("residual_saturation") = "0";
  options.set("residual_saturation").doc() = "Liquid's residual volume fraction";

  options.set_input("fluid_fraction") = VariableName(FORCES, "fluid_fraction");
  options.set("fluid_fraction").doc() = "Volume fraction of the fluid";

  options.set_parameter<TensorName<Scalar>>("max_fraction") = "1";
  options.set("max_fraction").doc() = "Maximum allowable volume fraction of the fluid";

  options.set_output("effective_saturation") = VariableName(STATE, "effective_saturation");
  options.set("effective_saturation").doc() = "Effective saturation";

  return options;
}

EffectiveSaturation::EffectiveSaturation(const OptionSet & options)
  : Model(options),
    _Sr(declare_parameter<Scalar>("Sr", "residual_saturation")),
    _phi(declare_input_variable<Scalar>("fluid_fraction")),
    _phimax(declare_parameter<Scalar>("phi_max", "max_fraction", /*allow_nonlinear=*/true)),
    _S(declare_output_variable<Scalar>("effective_saturation"))
{
}

void
EffectiveSaturation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
  {
    _S = (_phi / _phimax - _Sr) / (1.0 - _Sr);
  }

  if (dout_din)
  {
    if (_phi.is_dependent())
      _S.d(_phi) = 1.0 / (_phimax * (1 - _Sr));

    if (const auto * const phimax = nl_param("phi_max"))
      _S.d(*phimax) = -_phi / (_phimax * _phimax * (1 - _Sr));
  }
}
}
