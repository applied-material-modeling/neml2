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

#include "neml2/models/solid_mechanics/crystal_plasticity/FixOrientation.h"

#include "neml2/tensors/Rot.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(FixOrientation);

OptionSet
FixOrientation::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() =
      "Checks the value of the modified Rodrigues parameter by checking if "
      "\\f$ \\left\\lVert r \\right\\rVert > t \\f$, with \\f$ t \\f$ a threshold value set "
      "to 1.0 by default and replacing all the orientations that exceed this limit "
      "with their shadow parameters values.";

  options.set_input("input_orientation") = VariableName(STATE, "orientation");
  options.set("input_orientation").doc() = "Name of input tensor of orientations to operate on.";

  options.set_output("output_orientation") = VariableName(STATE, "orientation");
  options.set("output_orientation").doc() = "Name of output tensor";

  options.set<double>("threshold") = 1.0;
  options.set("threshold").doc() = "Threshold value for translating to the shadow parameters";

  return options;
}

FixOrientation::FixOrientation(const OptionSet & options)
  : Model(options),
    _output(declare_output_variable<Rot>("output_orientation")),
    _input(declare_input_variable<Rot>("input_orientation")),
    _threshold(options.get<double>("threshold"))
{
}

void
FixOrientation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _output = where(
        (Rot(_input).norm_sq() < _threshold).unsqueeze(-1), Rot(_input), Rot(_input).shadow());

  if (dout_din)
    if (_input.is_dependent())
    {
      const auto I = R2::identity(_input.options());
      _output.d(_input) = where((Rot(_input).norm_sq() < _threshold).unsqueeze(-1).unsqueeze(-1),
                                I,
                                Rot(_input).dshadow());
    }
}
} // namespace neml2
