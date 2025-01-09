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
#include "neml2/misc/assertions.h"

namespace neml2
{
OptionSet
SolidMechanicsDriver::expected_options()
{
  OptionSet options = TransientDriver::expected_options();
  options.doc() = "Driver for solid mechanics material model with optional thermal coupling.";

  EnumSelection control_selection({"STRAIN", "STRESS", "MIXED"}, "STRAIN");
  options.set<EnumSelection>("control") = control_selection;
  options.set("control").doc() =
      "External control of the material update. Options are " + control_selection.candidates_str();

  options.set<VariableName>("temperature") = VariableName(FORCES, "T");
  options.set("temperature").doc() = "Name of temperature";
  options.set<TensorName>("prescribed_temperature");
  options.set("prescribed_temperature").doc() =
      "Actual prescibed temperature values, when providing temperatures to the model";

  options.set<VariableName>("mixed_driving_force") = VariableName(FORCES, "fixed_values");
  options.set("mixed_driving_force").doc() = "Name of mixed driving force when using mixed control";
  options.set<TensorName>("prescribed_mixed_driving_force");
  options.set("prescribed_mixed_driving_force").doc() =
      "The fixed, controlled values provided as user input for the mixed control case.  Where the "
      "control signal is 0 these are strain/deformation values, where it is 1 these are stress "
      "values";

  options.set<VariableName>("mixed_control_signal") = VariableName(FORCES, "control");
  options.set("mixed_control_signal").doc() =
      "The name of the control signal for mixed control on the input axis";
  options.set<TensorName>("prescribed_mixed_control_signal");
  options.set("prescribed_mixed_control_signal").doc() =
      "The actual values of the control signal for mixed control. 0 implies strain/deformation "
      "control, 1 implies stress control";

  return options;
}

SolidMechanicsDriver::SolidMechanicsDriver(const OptionSet & options)
  : TransientDriver(options),
    _control(options.get<EnumSelection>("control")),
    _temperature_prescribed(options.get("prescribed_temperature").user_specified())
{
}

void
SolidMechanicsDriver::setup()
{
  TransientDriver::setup();

  if (_control == "STRAIN")
    init_strain_control(input_options());
  else if (_control == "STRESS")
    init_stress_control(input_options());
  else if (_control == "MIXED")
    init_mixed_control(input_options());
  else
    // LCOV_EXCL_START
    throw NEMLException("Unsupported control type.");
  // LCOV_EXCL_STOP

  if (_temperature_prescribed)
    init_temperature_control(input_options());
}

void
SolidMechanicsDriver::init_mixed_control(const OptionSet & options)
{
  _driving_force_name = options.get<VariableName>("mixed_driving_force");
  _driving_force = SR2(options.get<TensorName>("prescribed_mixed_driving_force"));
  _driving_force = _driving_force.to(_device);

  _mixed_control_name = options.get<VariableName>("mixed_control_signal");
  _mixed_control = SR2(options.get<TensorName>("prescribed_mixed_control_signal"));
  _mixed_control = _mixed_control.to(_device);
}

void
SolidMechanicsDriver::init_temperature_control(const OptionSet & options)
{
  _temperature_name = options.get<VariableName>("temperature");
  _temperature = Scalar(options.get<TensorName>("prescribed_temperature"));
  _temperature = _temperature.to(_device);
}

void
SolidMechanicsDriver::diagnose() const
{
  TransientDriver::diagnose();

  diagnostic_assert(_driving_force.batch_dim() >= 1,
                    "Input driving force (strain, stress, or mixed conditions) should have at "
                    "least one batch dimension for time steps but instead has batch dimension ",
                    _driving_force.batch_dim());

  diagnostic_assert(_time.batch_size(0) == _driving_force.batch_size(0),
                    "Input driving force (strain, stress, or mixed conditions) and time should "
                    "have the same number of time steps. The input time has ",
                    _time.batch_size(0),
                    " time steps, while the input driving force has ",
                    _driving_force.batch_size(0),
                    " time steps");

  if (_temperature_prescribed)
  {
    diagnostic_assert(_temperature.batch_dim() >= 1,
                      "Input temperature should have at least one batch dimension for time steps "
                      "but instead has batch dimension ",
                      _temperature.batch_dim());

    diagnostic_assert(
        _time.batch_size(0) == _temperature.batch_size(0),
        "Input temperature and time should have the same number of time steps. The input time has ",
        _time.batch_size(0),
        " time steps, while the input temperature has ",
        _temperature.batch_size(0),
        " time steps");
  }

  if (_control == "MIXED")
  {
    diagnostic_assert(_mixed_control.batch_dim() >= 1,
                      "Input control signal should have at least one batch dimension but instead "
                      "has batch dimension ",
                      _mixed_control.batch_dim());
    diagnostic_assert(
        _mixed_control.batch_size(0) == _time.batch_size(0),
        "Input control signal should have the same number of steps steps as time, but instead has ",
        _mixed_control.batch_size(0),
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
    _in[_mixed_control_name] = _mixed_control.batch_index({_step_count});
}
}
