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

#include "neml2/drivers/solid_mechanics/SolidMechanicsDriver.h"

namespace neml2
{
register_NEML2_object(SolidMechanicsDriver);

OptionSet
SolidMechanicsDriver::expected_options()
{
  OptionSet options = TransientDriver::expected_options();
  options.doc() =
      "Driver for small deformation solid mechanics material model with optional thermal coupling.";

  EnumSelection control_selection({"STRAIN", "STRESS", "MIXED"}, "STRAIN");
  options.set<EnumSelection>("control") = control_selection;
  options.set("control").doc() =
      "External control of the material update. Options are " + control_selection.candidates_str();

  options.set<VariableName>("total_strain") = VariableName("forces", "E");
  options.set("total_strain").doc() = "Total strain";

  options.set<VariableName>("cauchy_stress") = VariableName("forces", "S");
  options.set("cauchy_stress").doc() = "Cauchy stress";

  options.set<VariableName>("temperature") = VariableName("forces", "T");
  options.set("temperature").doc() = "Name of temperature";

  options.set<VariableName>("fixed_values") = VariableName("forces", "fixed_values");
  options.set("fixed_values").doc() = "Name of fixed values (when control = MIXED)";

  options.set<CrossRef<torch::Tensor>>("prescribed_strains");
  options.set("prescribed_strains").doc() = "Prescribed strain (when control = STRAIN)";

  options.set<CrossRef<torch::Tensor>>("prescribed_stresses");
  options.set("prescribed_stresses").doc() = "Prescribed stress (when control = STRESS)";

  options.set<CrossRef<torch::Tensor>>("prescribed_temperatures");
  options.set("prescribed_temperatures").doc() =
      "Actual prescibed temperature values, when providing temperatures to the model";

  options.set<CrossRef<torch::Tensor>>("prescribed_mixed_conditions");
  options.set("prescribed_mixed_conditions").doc() =
      "The fixed, controlled values provided as user input for the mixed control case.  Where the "
      "control signal is 0 these are strain values, where it is 1 these are stress values";

  options.set<VariableName>("control_name") = VariableName("forces", "control");
  options.set("control_name").doc() = "The name of the control signal on the input axis";

  options.set<CrossRef<torch::Tensor>>("prescribed_control");
  options.set("prescribed_control").doc() = "The actual values of the control signal.  0 implies "
                                            "strain control, 1 implies stress control";

  return options;
}

SolidMechanicsDriver::SolidMechanicsDriver(const OptionSet & options)
  : TransientDriver(options),
    _control(options.get<EnumSelection>("control")),
    _control_name(options.get<VariableName>("control_name")),
    _temperature_name(options.get<VariableName>("temperature")),
    _temperature_prescribed(
        !options.get<CrossRef<torch::Tensor>>("prescribed_temperatures").raw().empty()),
    _temperature(_temperature_prescribed
                     ? Scalar(options.get<CrossRef<torch::Tensor>>("prescribed_temperatures"))
                     : Scalar()),
    _control_signal(_control == "MIXED"
                        ? SR2(options.get<CrossRef<torch::Tensor>>("prescribed_control"))
                        : SR2())

{
  if (_control == "STRAIN")
  {
    _driving_force = SR2(options.get<CrossRef<torch::Tensor>>("prescribed_strains"));
    _driving_force_name = options.get<VariableName>("total_strain");
  }
  else if (_control == "STRESS")
  {
    _driving_force = SR2(options.get<CrossRef<torch::Tensor>>("prescribed_stresses"));
    _driving_force_name = options.get<VariableName>("cauchy_stress");
  }
  else if (_control == "MIXED")
  {
    _driving_force = SR2(options.get<CrossRef<torch::Tensor>>("prescribed_mixed_conditions"));
    _driving_force_name = options.get<VariableName>("fixed_values");
  }
  else
    // LCOV_EXCL_START
    throw NEMLException("Unsupported control type.");
  // LCOV_EXCL_STOP

  _driving_force = _driving_force.to(_device);

  if (_temperature_prescribed)
    _temperature = _temperature.to(_device);
}

void
SolidMechanicsDriver::diagnose(std::vector<Diagnosis> & diagnoses) const
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

  if (_temperature_prescribed)
  {
    diagnostic_assert(diagnoses,
                      _temperature.batch_dim() >= 1,
                      "Input temperature should have at least one batch dimension for time steps "
                      "but instead has batch dimension ",
                      _temperature.batch_dim());

    diagnostic_assert(
        diagnoses,
        _time.batch_size(0) == _temperature.batch_size(0),
        "Input temperature and time should have the same number of time steps. The input time has ",
        _time.batch_size(0),
        " time steps, while the input temperature has ",
        _temperature.batch_size(0),
        " time steps");
  }

  if (_control == "MIXED")
  {
    diagnostic_assert(diagnoses,
                      _control_signal.batch_dim() >= 1,
                      "Input control signal should have at least one batch dimension but instead "
                      "has batch dimension ",
                      _control_signal.batch_dim());
    diagnostic_assert(
        diagnoses,
        _control_signal.batch_size(0) == _time.batch_size(0),
        "Input control signal should have the same number of steps steps as time, but instead has ",
        _control_signal.batch_size(0),
        " time steps");
  }
}

void
SolidMechanicsDriver::update_forces()
{
  TransientDriver::update_forces();

  _in[_driving_force_name] = _driving_force.batch_index({_step_count});

  if (_temperature_prescribed)
    _in[_temperature_name] = _temperature.batch_index({_step_count});

  if (_control == "MIXED")
    _in[_control_name] = _control_signal.batch_index({_step_count});
}
}
