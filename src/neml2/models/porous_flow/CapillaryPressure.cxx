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

#include "neml2/models/porous_flow/CapillaryPressure.h"
#include "neml2/tensors/functions/log10.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
OptionSet
CapillaryPressure::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Relate the porous flow capillary pressure to the effective saturation";

  options.set_input("effective_saturation") = VariableName(STATE, "S");
  options.set("effective_saturation").doc() = "The effective saturation";

  options.set_output("capillary_pressure") = VariableName(STATE, "Pc");
  options.set("capillary_pressure").doc() = "Capillary pressure.";

  options.set<bool>("log_extension") = false;
  options.set("log_extension").doc() = "Whether to apply logarithmic extension";

  options.set<double>("transition_saturation") = 0.0;
  options.set("transition_saturation").doc() = "The transistion value of the effective saturation "
                                               "below which to apply the logarithmic extension";

  return options;
}

CapillaryPressure::CapillaryPressure(const OptionSet & options)
  : Model(options),
    _S(declare_input_variable<Scalar>("effective_saturation")),
    _Pc(declare_output_variable<Scalar>("capillary_pressure")),
    _log_extension(options.get<bool>("log_extension")),
    _Sp_specified(options.user_specified("transition_saturation")),
    _Sp(options.get<double>("transition_saturation"))
{
  if (_log_extension && !_Sp_specified)
    throw NEMLException("BrooksCoreyCapillaryPressure: log_extension is set to true, but "
                        "transition_saturation is not specified.");

  if (!_log_extension && _Sp_specified)
    throw NEMLException("BrooksCoreyCapillaryPressure: transition_saturation is specified, but "
                        "log_extension is not set to true.");

  if (_Sp < 0.0 || _Sp > 1.0)
    throw NEMLException(
        "BrooksCoreyCapillaryPressure: transition_saturation must be in the range [0, 1].");
}

void
CapillaryPressure::set_value(bool out, bool dout, bool d2out)
{
  auto [Pc, dPc_dS, d2Pc_dS2] = calculate_pressure(_S, out, dout, d2out);

  if (_log_extension)
  {
    // Apply logarithmic extension
    constexpr double ln10 = 2.302585092994;
    auto [Pcs, dPcs_dS, d2Pcs_dS2] =
        calculate_pressure(Scalar::full(_Sp, _S.options()), true, true, false);
    auto slope = 1.0 / (ln10 * Pcs) * dPcs_dS;
    auto yintercept = log10(Pcs) - slope * _Sp;
    auto Pc_ext = Scalar(pow(10, slope * _S + yintercept));

    if (out)
      _Pc = where(_S < _Sp, Pc_ext, Pc);

    if (dout)
      _Pc.d(_S) = where(_S < _Sp, (ln10 * slope) * Pc_ext, dPc_dS);

    if (d2out)
      _Pc.d(_S, _S) = where(_S < _Sp, (ln10 * slope) * (ln10 * slope) * Pc_ext, d2Pc_dS2);
  }
  else
  {
    if (out)
      _Pc = Pc;

    if (dout)
      _Pc.d(_S) = dPc_dS;

    if (d2out)
      _Pc.d(_S, _S) = d2Pc_dS2;
  }
}

} // namespace neml2
