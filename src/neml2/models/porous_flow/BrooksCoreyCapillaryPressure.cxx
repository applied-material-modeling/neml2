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

#include "neml2/models/porous_flow/BrooksCoreyCapillaryPressure.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(BrooksCoreyCapillaryPressure);
OptionSet
BrooksCoreyCapillaryPressure::expected_options()
{
  OptionSet options = CapillaryPressure::expected_options();
  options.doc() +=
      " using the Brooks Corey correlation taking the form of \\f$ P_c = P_t S_e^{-\\frac{1}{p}} "
      "\\f$. Here \\f$ S_e \\f$ is the effective saturation,\\f$ P_t \\f$ is the threshold "
      "pressure at zero saturation, and \\f$ p \\f$ is the shape parameter";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("threshold_pressure");
  options.set("threshold_pressure").doc() = "The threshold entry pressure";

  options.set_parameter<TensorName<Scalar>>("exponent");
  options.set("exponent").doc() = "The shape parameter p";

  return options;
}

BrooksCoreyCapillaryPressure::BrooksCoreyCapillaryPressure(const OptionSet & options)
  : CapillaryPressure(options),
    _Pt(declare_parameter<Scalar>("threshold", "threshold_pressure")),
    _p(declare_parameter<Scalar>("p", "exponent"))
{
}

std::tuple<Scalar, Scalar, Scalar>
BrooksCoreyCapillaryPressure::calculate_pressure(const Scalar & S,
                                                 bool out,
                                                 bool dout_din,
                                                 bool d2out_din2) const
{
  auto Pc = out ? _Pt * pow(S, -1.0 / _p) : Scalar();
  auto dPc_dS = dout_din ? -_Pt / _p * pow(S, -1.0 / _p - 1.0) : Scalar();
  auto d2Pc_dS2 = d2out_din2 ? -_Pt / _p * (-1.0 / _p - 1.0) * pow(S, -1.0 / _p - 2.0) : Scalar();

  return std::make_tuple(Pc, dPc_dS, d2Pc_dS2);
}
}
