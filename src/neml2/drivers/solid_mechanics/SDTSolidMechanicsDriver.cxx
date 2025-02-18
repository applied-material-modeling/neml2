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

#include "neml2/drivers/solid_mechanics/SDTSolidMechanicsDriver.h"

namespace neml2
{
register_NEML2_object(SDTSolidMechanicsDriver);

OptionSet
SDTSolidMechanicsDriver::expected_options()
{
  OptionSet options = SolidMechanicsDriver::expected_options();
  options.doc() +=
      " This driver is specialized for small deformation models using the total formulation.";

  options.set<VariableName>("strain") = VariableName(FORCES, "E");
  options.set("strain").doc() = "Name of the strain used to drive the update";
  options.set<TensorName>("prescribed_strain");
  options.set("prescribed_strain").doc() = "Prescribed strain (when control = STRAIN)";

  options.set<VariableName>("stress") = VariableName(FORCES, "S");
  options.set("stress").doc() = "Name of the stress used to drive the update";
  options.set<TensorName>("prescribed_stress");
  options.set("prescribed_stress").doc() = "Prescribed stress (when control = STRESS)";

  return options;
}

SDTSolidMechanicsDriver::SDTSolidMechanicsDriver(const OptionSet & options)
  : SolidMechanicsDriver(options)
{
}

void
SDTSolidMechanicsDriver::init_strain_control(const OptionSet & options)
{
  _driving_force_name = options.get<VariableName>("strain");
  _driving_force = SR2(options.get<TensorName>("prescribed_strain"));
  _driving_force = _driving_force.to(_device);
}

void
SDTSolidMechanicsDriver::init_stress_control(const OptionSet & options)
{
  _driving_force_name = options.get<VariableName>("stress");
  _driving_force = SR2(options.get<TensorName>("prescribed_stress"));
  _driving_force = _driving_force.to(_device);
}
} // namespace neml2
