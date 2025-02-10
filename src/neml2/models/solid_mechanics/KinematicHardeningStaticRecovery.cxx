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

#include "neml2/models/solid_mechanics/KinematicHardeningStaticRecovery.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
OptionSet
KinematicHardeningStaticRecovery::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() += "This object defines kinematic hardening static recovery on a backstress term.";

  options.set_input("back_stress") = VariableName(STATE, "internal", "X");
  options.set("back_stress").doc() = "Back stress";

  options.set_output("back_stress_rate");
  options.set("back_stress_rate").doc() =
      "Back stress rate, defaults to back_stress + _recovery_rate";

  return options;
}

KinematicHardeningStaticRecovery::KinematicHardeningStaticRecovery(const OptionSet & options)
  : Model(options),
    _X(declare_input_variable<SR2>("back_stress")),
    _X_dot(declare_output_variable<SR2>(options.get<VariableName>("back_stress_rate").empty()
                                            ? _X.name().with_suffix("_recovery_rate")
                                            : options.get<VariableName>("back_stress_rate")))
{
}

} // namespace neml2
