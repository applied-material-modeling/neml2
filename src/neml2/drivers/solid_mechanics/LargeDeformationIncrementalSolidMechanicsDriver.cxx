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

#include "neml2/drivers/solid_mechanics/LargeDeformationIncrementalSolidMechanicsDriver.h"

namespace neml2
{
register_NEML2_object(LargeDeformationIncrementalSolidMechanicsDriver);

OptionSet
LargeDeformationIncrementalSolidMechanicsDriver::expected_options()
{
  OptionSet options = TransientDriver::expected_options();
  options.doc() = "Driver for large deformation solid mechanics material model. The material model "
                  "is updated *incrementally*.";

  options.set<std::string>("control") = "STRAIN";
  options.set("control").doc() = "External control of the material update. Options are STRAIN and "
                                 "STRESS, for strain control and stress control, respectively.";

  options.set<VariableName>("deformation_rate") = VariableName("forces", "deformation_rate");
  options.set("deformation_rate").doc() = "Deformation rate";

  options.set<VariableName>("cauchy_stress_rate") = VariableName("forces", "cauchy_stress_rate");
  options.set("cauchy_stress_rate").doc() = "Cauchy stress rate";

  options.set<VariableName>("vorticity") = VariableName("forces", "vorticity");
  options.set("vorticity").doc() = "Vorticity";

  options.set<CrossRef<torch::Tensor>>("prescribed_deformation_rate");
  options.set("prescribed_deformation_rate").doc() =
      "Prescribed deformation rate (when control = STRAIN)";

  options.set<CrossRef<torch::Tensor>>("prescribed_cauchy_stress_rate");
  options.set("prescribed_cauchy_stress_rate").doc() =
      "Prescribed cauchy stress rate (when control = STRESS)";

  options.set<CrossRef<torch::Tensor>>("prescribed_vorticity");
  options.set("prescribed_vorticity").doc() = "Prescribed vorticity";

  return options;
}

LargeDeformationIncrementalSolidMechanicsDriver::LargeDeformationIncrementalSolidMechanicsDriver(
    const OptionSet & options)
  : TransientDriver(options),
    _control(options.get<std::string>("control")),
    _vorticity_name(options.get<VariableName>("vorticity")),
    _vorticity_prescribed(
        !options.get<CrossRef<torch::Tensor>>("prescribed_vorticity").raw().empty()),
    _vorticity(_vorticity_prescribed
                   ? WR2(options.get<CrossRef<torch::Tensor>>("prescribed_vorticity"))
                   : WR2::zeros(_driving_force.batch_sizes()))
{
  if (_control == "STRAIN")
  {
    _driving_force = SR2(options.get<CrossRef<torch::Tensor>>("prescribed_deformation_rate"));
    _driving_force_name = options.get<VariableName>("deformation_rate");
  }
  else if (_control == "STRESS")
  {
    _driving_force = SR2(options.get<CrossRef<torch::Tensor>>("prescribed_cauchy_stress_rate"));
    _driving_force_name = options.get<VariableName>("cauchy_stress_rate");
  }
  else
    // LCOV_EXCL_START
    throw NEMLException("Unsupported control type.");
  // LCOV_EXCL_STOP

  _driving_force = _driving_force.to(_device);
  _vorticity = _vorticity.to(_device);
}

void
LargeDeformationIncrementalSolidMechanicsDriver::diagnose(std::vector<Diagnosis> & diagnoses) const
{
  TransientDriver::diagnose(diagnoses);

  diagnostic_assert(diagnoses,
                    _driving_force.batch_dim() >= 1,
                    "Input driving force (strain, stress, or mixed conditions) should have at "
                    "least one batch dimension for time steps but instead has batch dimension ",
                    _driving_force.batch_dim());

  diagnostic_assert(diagnoses,
                    _time.batch_size(0) == _driving_force.batch_size(0),
                    "Input driving force (strain, stress, or mixed conditions) and time should "
                    "have the same number of time steps. The input time has ",
                    _time.batch_size(0),
                    " time steps, while the input driving force has ",
                    _driving_force.batch_size(0),
                    " time steps");

  diagnostic_assert(
      diagnoses,
      _time.batch_size(0) == _vorticity.batch_size(0),
      "Input vorticity and time should have the same number of time steps. The input time has ",
      _time.batch_size(0),
      " time steps, while the input driving force has ",
      _vorticity.batch_size(0),
      " time steps");
}

void
LargeDeformationIncrementalSolidMechanicsDriver::update_forces()
{
  TransientDriver::update_forces();
  _in[_driving_force_name] = _driving_force.batch_index({_step_count});
  _in[_vorticity_name] = _vorticity.batch_index({_step_count});
}
}
