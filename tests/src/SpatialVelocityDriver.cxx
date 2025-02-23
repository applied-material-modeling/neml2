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

#include "SpatialVelocityDriver.h"
#include "neml2/misc/assertions.h"

namespace neml2
{

register_NEML2_object(SpatialVelocityDriver);

OptionSet
SpatialVelocityDriver::expected_options()
{
  OptionSet options = TransientDriver::expected_options();
  options.doc() =
      "Driver for solid mechanics material model driven by an R2 spatial velocity gradient.";

  options.set<VariableName>("spatial_velocity_gradient") =
      VariableName(FORCES, "spatial_velocity_gradient");
  options.set("spatial_velocity_gradient").doc() = "Spatial velocity gradient";
  options.set<TensorName<R2>>("prescribed_spatial_velocity_gradient");
  options.set("prescribed_spatial_velocity_gradient").doc() =
      "Prescribed spatial velocity gradient";

  return options;
}

SpatialVelocityDriver::SpatialVelocityDriver(const OptionSet & options)
  : TransientDriver(options),
    _driving_force_name(options.get<VariableName>("spatial_velocity_gradient")),
    _driving_force(options.get<TensorName<R2>>("prescribed_spatial_velocity_gradient").resolve())
{
}

void
SpatialVelocityDriver::diagnose() const
{
  TransientDriver::diagnose();

  diagnostic_assert(_driving_force.batch_dim() >= 1,
                    "Input spatial velocity gradient should have at "
                    "least one batch dimension for time steps but instead has batch dimension ",
                    _driving_force.batch_dim());

  diagnostic_assert(_time.batch_size(0) == _driving_force.batch_size(0),
                    "Input spatial velocity gradient and time should "
                    "have the same number of time steps. The input time has ",
                    _time.batch_size(0),
                    " time steps, while the input driving force has ",
                    _driving_force.batch_size(0),
                    " time steps");
}

void
SpatialVelocityDriver::update_forces()
{
  TransientDriver::update_forces();

  _in[_driving_force_name] = _driving_force.batch_index({_step_count});
}
}
