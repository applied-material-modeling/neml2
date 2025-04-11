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

#include "neml2/models/reactive_infiltration/VanGenuchtenPressure.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(VanGenuchtenPressure);
OptionSet
VanGenuchtenPressure::expected_options()
{
  OptionSet options = PorousFlowCapillaryPressure::expected_options();
  options.doc() = "Define the van Genuchten porous flow capillary pressure, takes the form of \\f$ "
                  "\\frac{1}{a} "
                  "\\frac{S_e^{-\\frac{1}{m}} - 1}{1-m} \\f$ for \\f$ S_e < 1 \\f$ "
                  "and 0 everywhere else. Here \\f$ S_e \\f$ is the effective saturation,\\f$ a, m "
                  "\\f$ is the fitting parameter";

  options.set_parameter<TensorName<Scalar>>("scaling_constant");
  options.set("scaling_constant").doc() = "scaling reciprocal constant, a";

  options.set_parameter<TensorName<Scalar>>("power");
  options.set("power").doc() = "power, m";
#include "neml2/tensors/functions/heaviside.h"
  return options;
}

VanGenuchtenPressure::VanGenuchtenPressure(const OptionSet & options)
  : PorousFlowCapillaryPressure(options),
    _a(declare_parameter<Scalar>("a", "scaling_constant")),
    _m(declare_parameter<Scalar>("m", "power"))
{
}

void
VanGenuchtenPressure::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _Pc = 1 / _a * pow((pow(_S, -1.0 / _m) - 1.0), 1 - _m) * (1 - heaviside(_S - 1.0));
  }

  if (dout_din)
  {
    _Pc.d(_S) = -1 / (_a * _m) * (1 - _m) * pow(pow(_S, -1 / _m) - 1, -_m) * pow(_S, -1 / _m - 1) *
                (1 - heaviside(_S - 1.0));
  }
}
}