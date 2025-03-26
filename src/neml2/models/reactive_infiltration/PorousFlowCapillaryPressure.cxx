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

#include "neml2/models/reactive_infiltration/PorousFlowCapillaryPressure.h"

namespace neml2
{
OptionSet
PorousFlowCapillaryPressure::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Relate the porous flow capillary pressure to the effective saturation";

  options.set_input("effective_saturation") = VariableName(STATE, "effective_saturation");
  options.set("effective_saturation").doc() = "The effective saturation";

  options.set_output("capillary_pressure") = VariableName(STATE, "capillary_pressure");
  options.set("capillary_pressure").doc() = "Porous flow capillary pressure.";

  return options;
}

PorousFlowCapillaryPressure::PorousFlowCapillaryPressure(const OptionSet & options)
  : Model(options),
    _S(declare_input_variable<Scalar>("effective_saturation")),
    _Pc(declare_output_variable<Scalar>("capillary_pressure"))
{
}

} // namespace neml2