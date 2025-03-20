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

#include "neml2/drivers/pyrolysis/PyrolysisDriver.h"

namespace neml2
{
register_NEML2_object(PyrolysisDriver);

OptionSet
PyrolysisDriver::expected_options()
{
  OptionSet options = TransientDriver::expected_options();
  options.doc() = "Driver for the pyrolysis process";

  options.set<TensorName<Scalar>>("prescribed_temperature");
  options.set("prescribed_temperature").doc() =
      "Actual prescibed temperature values, when providing temperatures to the model";

  options.set<VariableName>("temperature") = VariableName("forces", "T");
  options.set("temperature").doc() = "Name of the variable for temperature.";

  return options;
}

PyrolysisDriver::PyrolysisDriver(const OptionSet & options)
  : TransientDriver(options),
    _temperature(options.get<TensorName<Scalar>>("prescribed_temperature").resolve()),
    _temperature_name(options.get<VariableName>("temperature"))
{
  _temperature = _temperature.to(_device);
}

void
PyrolysisDriver::diagnose() const
{
  TransientDriver::diagnose();
  diagnostic_assert(_temperature.batch_dim() >= 1,
                    "prescribed_temperature should have at least one batch dimension but instead "
                    "has batch dimension ",
                    _temperature.batch_dim());
  diagnostic_assert(
      _temperature.batch_size(0) == _time.batch_size(0),
      "prescribed_temperature should have the same number of steps as time, but instead has ",
      _temperature.batch_size(0),
      " steps");
}

void
PyrolysisDriver::update_forces()
{
  TransientDriver::update_forces();
  _in[_temperature_name] = _temperature.batch_index({_step_count});
}
}
