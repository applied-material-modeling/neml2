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

#include "neml2/models/porous_flow/VanGenuchtenCapillaryPressure.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/log10.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(VanGenuchtenCapillaryPressure);
OptionSet
VanGenuchtenCapillaryPressure::expected_options()
{
  OptionSet options = CapillaryPressure::expected_options();
  options.doc() +=
      " using the van Genuchten correlation for capillary pressure, taking the form of \\f$ "
      "a \\left( S_e^{-\\frac{1}{m}} - 1 \\right)^{1-m} \\f$. Here \\f$ S_e \\f$ is the "
      "effective saturation,\\f$ a \\f$ and \\f$ m \\f$ are shape parameters";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("a");
  options.set("a").doc() = "Shape parameter a";

  options.set_parameter<TensorName<Scalar>>("m");
  options.set("m").doc() = "Shape parameter m";

  return options;
}

VanGenuchtenCapillaryPressure::VanGenuchtenCapillaryPressure(const OptionSet & options)
  : CapillaryPressure(options),
    _a(declare_parameter<Scalar>("a", "a")),
    _m(declare_parameter<Scalar>("m", "m"))
{
}

std::tuple<Scalar, Scalar, Scalar>
VanGenuchtenCapillaryPressure::calculate_pressure(const Scalar & S,
                                                  bool out,
                                                  bool dout_din,
                                                  bool d2out_din2) const
{
  const auto eps = machine_precision(S.scalar_type()).toDouble();
  auto Sc = where(S > (1.0 - eps), Scalar::ones_like(S) - eps, S);

  auto f_s = _a * pow((pow(Sc, -1.0 / _m) - 1.0), 1 - _m);
  auto dfds_s = -_a / _m * (1 - _m) * pow(pow(Sc, -1 / _m) - 1, -_m) * pow(Sc, -1 / _m - 1);
  auto d2fds2_s = -_a * ((_m - 1) * (_m * pow(Sc, (1 / _m)) + pow(Sc, (1 / _m)) - 1)) /
                  (_m * _m * pow(Sc, (1 / _m + 2)) * pow((1 / pow(Sc, (1 / _m)) - 1), _m) *
                   (pow(Sc, (1 / _m)) - 1));

  return std::make_tuple(
      out ? f_s : Scalar(), dout_din ? dfds_s : Scalar(), d2out_din2 ? d2fds2_s : Scalar());
}
}
