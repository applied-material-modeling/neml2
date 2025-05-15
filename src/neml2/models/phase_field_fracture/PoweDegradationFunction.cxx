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

#include "neml2/models/phase_field_fracture/PowerDegradationFunction.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/Scalar.h"


namespace neml2
{
register_NEML2_object(PowerDegradationFunction);

OptionSet
PowerDegradationFunction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Power degradation function to degrade the elastic strain energy density";

  options.set_input("damage") = VariableName(STATE, "d");
  options.set<TensorName<Scalar>>("power");
  options.set_output("degradation") = VariableName(STATE, "g");

  return options;
}

PowerDegradationFunction::PowerDegradationFunction(const OptionSet & options)
  : Model(options),
    _d(declare_input_variable<Scalar>("damage")),
    _p(declare_parameter<Scalar>("p", "power")),
    _g(declare_output_variable<Scalar>("degradation"))
{
}

void
PowerDegradationFunction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
  {
    _g = pow((1 - _d), _p);
    
  }

  if (dout_din)
  {
    _g.d(_d) = _p * pow((1 - _d), (_p - 1)) * (- _d);
  }

  if (d2out_din2)
  {
    _g.d(_d, _d) = (_p * _p) * pow((1 - _d), (_p - 2)) * (- _d) * (- _d) + _p * pow((1 - _d), (_p - 1)) * (- 1);
  }
}
} // namespace neml2
