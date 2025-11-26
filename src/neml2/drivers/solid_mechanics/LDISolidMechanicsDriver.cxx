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

#include "neml2/drivers/solid_mechanics/LDISolidMechanicsDriver.h"
#include "neml2/misc/assertions.h"
#include "neml2/base/LabeledAxis.h"
#include "neml2/models/Model.h"

namespace neml2
{
register_NEML2_object(LDISolidMechanicsDriver);

OptionSet
LDISolidMechanicsDriver::expected_options()
{
  OptionSet options = SolidMechanicsDriver::expected_options();
  options.doc() +=
      " This driver is specialized for large deformation models using the incremental formulation.";

  options.set<VariableName>("deformation_rate") = VariableName(FORCES, "deformation_rate");
  options.set("deformation_rate").doc() = "Deformation rate";
  options.set<TensorName<SR2>>("prescribed_deformation_rate");
  options.set("prescribed_deformation_rate").doc() =
      "Prescribed deformation rate (when control = STRAIN)";

  options.set<VariableName>("cauchy_stress_rate") = VariableName(FORCES, "cauchy_stress_rate");
  options.set("cauchy_stress_rate").doc() = "Cauchy stress rate";
  options.set<TensorName<SR2>>("prescribed_cauchy_stress_rate");
  options.set("prescribed_cauchy_stress_rate").doc() =
      "Prescribed cauchy stress rate (when control = STRESS)";

  options.set<VariableName>("vorticity") = VariableName(FORCES, "vorticity");
  options.set("vorticity").doc() = "Vorticity";
  options.set<TensorName<WR2>>("prescribed_vorticity");
  options.set("prescribed_vorticity").doc() = "Prescribed vorticity";

  options.set<bool>("cp_warmup") = false;
  options.set("cp_warmup").doc() =
      "Whether to perform a warm-up step for crystal plasticity models. The warm-up step uses a "
      "relaxed/damped elastic predictor for the very first time step.";
  options.set<double>("cp_warmup_elastic_scale") = 1.0;
  options.set("cp_warmup_elastic_scale").doc() =
      "Elastic step scale factor used for the crystal plasticity warm-up step";
  options.set<VariableName>("cp_warmup_elastic_strain") = VariableName(STATE, "elastic_strain");
  options.set("cp_warmup_elastic_strain").doc() =
      "Elastic strain name used for the CP warm-up step";

  return options;
}

LDISolidMechanicsDriver::LDISolidMechanicsDriver(const OptionSet & options)
  : SolidMechanicsDriver(options),
    _vorticity_prescribed(options.get("prescribed_vorticity").user_specified()),
    _cp_warmup(options.get<bool>("cp_warmup")),
    _cp_warmup_elastic_scale(options.get<double>("cp_warmup_elastic_scale")),
    _cp_warmup_elastic_strain(options.get<VariableName>("cp_warmup_elastic_strain"))
{
}

void
LDISolidMechanicsDriver::setup()
{
  SolidMechanicsDriver::setup();
  init_vorticity_control(input_options());
}

void
LDISolidMechanicsDriver::init_strain_control(const OptionSet & options)
{
  _driving_force_name = options.get<VariableName>("deformation_rate");
  _driving_force = resolve_tensor<SR2>("prescribed_deformation_rate");
  _driving_force = _driving_force.to(_device);
}

void
LDISolidMechanicsDriver::init_stress_control(const OptionSet & options)
{
  _driving_force_name = options.get<VariableName>("cauchy_stress_rate");
  _driving_force = resolve_tensor<SR2>("prescribed_cauchy_stress_rate");
  _driving_force = _driving_force.to(_device);
}

void
LDISolidMechanicsDriver::init_vorticity_control(const OptionSet & options)
{
  _vorticity_name = options.get<VariableName>("vorticity");
  _vorticity = resolve_tensor<WR2>("prescribed_vorticity");
  _vorticity = _vorticity.to(_device);
}

void
LDISolidMechanicsDriver::diagnose() const
{
  SolidMechanicsDriver::diagnose();

  if (_vorticity_prescribed)
  {
    diagnostic_assert(_vorticity.dynamic_dim() >= 1,
                      "Input vorticity should have at least one batch dimension but instead "
                      "has batch dimension ",
                      _vorticity.dynamic_dim());
    diagnostic_assert(

        _vorticity.dynamic_size(0) == _time.dynamic_size(0),
        "Input vorticity should have the same number of steps steps as time, but instead has ",
        _vorticity.dynamic_size(0),
        " time steps");
  }

  if (_cp_warmup)
  {
    diagnostic_assert(_control == "STRAIN", "CP warm-up step is only supported for STRAIN control");
    diagnostic_assert(_model->input_variables().count(_cp_warmup_elastic_strain),
                      "Model's input should have variable ",
                      _cp_warmup_elastic_strain,
                      " for the CP warm-up step but it does not");
  }
}

void
LDISolidMechanicsDriver::update_forces()
{
  SolidMechanicsDriver::update_forces();
  _in[_vorticity_name] = _vorticity.dynamic_index({_step_count});
}

void
LDISolidMechanicsDriver::apply_predictor()
{
  SolidMechanicsDriver::apply_predictor();

  if (_cp_warmup && (_step_count == 1))
  {
    const auto D = SR2(_in[_driving_force_name]);
    const auto t = Scalar(_in[_time_name]);
    const auto t_n = Scalar(_result_in[_step_count - 1][_time_name]);
    _in[_cp_warmup_elastic_strain] = D * (t - t_n) * _cp_warmup_elastic_scale;
  }
}

} // namespace neml2
