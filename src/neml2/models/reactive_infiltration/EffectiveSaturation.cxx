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

#include "neml2/models/reactive_infiltration/EffectiveSaturation.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(EffectiveSaturation);
OptionSet
EffectiveSaturation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the effective saturation (volume fraction), takes the form of \\f$ S = "
      "\\frac{\\phi^* - S_r}{1-S_r} \\f$ where "
      "\\f$ \\phi \\f$ is the flow species volume fraction,\\f$ \\phi_{max} \\f$ is the maximum "
      "allowable flow species volume fraction and \\f$ "
      "S_r \\f$ is the residual liquid volume fraction";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("residual_volume_fraction") = {TensorName<Scalar>("0")};
  options.set("residual_volume_fraction").doc() = "Liquid's residual volume fraction";

  options.set_input("flow_fraction") = VariableName(FORCES, "flow_fraction");
  options.set("flow_fraction").doc() = "Volume fraction of the flow (liquid or gas) phase";

  options.set_input("max_fraction") = VariableName(STATE, "max_fraction");
  options.set("max_fraction").doc() =
      "Maximum allowable volume fraction of the flow (liquid or gas) phase";

  options.set_output("effective_saturation") = VariableName(STATE, "effective_saturation");
  options.set("effective_saturation").doc() = "Effective saturation";

  return options;
}

EffectiveSaturation::EffectiveSaturation(const OptionSet & options)
  : Model(options),
    _Sr(declare_parameter<Scalar>("Sr", "residual_volume_fraction")),
    _phi(declare_input_variable<Scalar>("flow_fraction")),
    _phimax(declare_input_variable<Scalar>("max_fraction")),
    _S(declare_output_variable<Scalar>("effective_saturation"))
{
}

void
EffectiveSaturation::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto val = (_phi / _phimax - _Sr) / (1.0 - _Sr);

  if (out)
  {
    _S = val;
    // where(val >= 1, Scalar::ones_like(_phi), val); // clamp(val, 0.0, 1.0);
  }

  if (dout_din)
  {
    // _S.d(_phi) = where(
    //     at::logical_or(val >= 1, val <= 0), Scalar::zeros_like(_phi), 1.0 / (_phimax * (1 -
    //     _Sr)));
    _S.d(_phi) = 1.0 / (_phimax * (1 - _Sr));
    // where(val >= 1, Scalar::zeros_like(_phi), 1.0 / (_phimax * (1 - _Sr)));
    _S.d(_phimax) = -_phi / (_phimax * _phimax * (1 - _Sr));
    // where(val >= 1, Scalar::zeros_like(_phi), -_phi / (_phimax * _phimax * (1 - _Sr)));
  }

  if (d2out_din2)
  {
    _S.d(_phi, _phimax) = -1.0 / ((1 - _Sr) * _phimax * _phimax);
    // where(val >= 1, Scalar::zeros_like(_phi), -1.0 / ((1 - _Sr) * _phimax * _phimax));
    _S.d(_phimax, _phi) = -1.0 / (_phimax * _phimax * (1 - _Sr));
    //  where(val >= 1, Scalar::zeros_like(_phi), -1.0 / (_phimax * _phimax * (1 - _Sr)));
    _S.d(_phimax, _phimax) = 2.0 * _phi / (_phimax * _phimax * _phimax * (1 - _Sr));
    // where(val >= 1, Scalar::zeros_like(_phi), 2.0 * _phi / (_phimax * _phimax * _phimax * (1 -
    // _Sr))); 0 otherwise
  }
}
}