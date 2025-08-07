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

#include "neml2/models/solid_mechanics/FredrickArmstrongPlasticHardening.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/norm.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(FredrickArmstrongPlasticHardening);

OptionSet
FredrickArmstrongPlasticHardening::expected_options()
{
  OptionSet options = FlowRule::expected_options();
  options.doc() +=
      " This object defines the non-associative Fredrick-Armstrong kinematic hardening. In the "
      "model, back stress is directly treated as an internal variable. Rate of back stress is "
      "given as \\f$ \\dot{\\boldsymbol{X}} = \\left( \\frac{2}{3} C \\frac{\\partial f}{\\partial "
      "\\boldsymbol{M}} - g \\boldsymbol{X} \\right) \\dot{\\gamma} \\f$."
      "\\f$ \\frac{\\partial f}{\\partial \\boldsymbol{M}} \\f$ is the flow "
      "direction, \\f$ \\dot{\\gamma} \\f$ is the flow rate, and \\f$ C \\f$ and \\f$ g \\f$ are "
      "material parameters.";

  options.set_input("back_stress") = VariableName(STATE, "internal", "X");
  options.set("back_stress").doc() = "Back stress";

  options.set_output("back_stress_rate");
  options.set("back_stress_rate").doc() = "Back stress rate, defaults to back_stress + _rate";

  options.set_input("flow_direction") = VariableName(STATE, "internal", "NM");
  options.set("flow_direction").doc() = "Flow direction";

  options.set_parameter<TensorName<Scalar>>("C");
  options.set("C").doc() = "Kinematic hardening coefficient";

  options.set_parameter<TensorName<Scalar>>("g");
  options.set("g").doc() = "Dynamic recovery coefficient";

  return options;
}

FredrickArmstrongPlasticHardening::FredrickArmstrongPlasticHardening(const OptionSet & options)
  : FlowRule(options),
    _X(declare_input_variable<SR2>("back_stress")),
    _NM(declare_input_variable<SR2>("flow_direction")),
    _X_dot(declare_output_variable<SR2>(options.get<VariableName>("back_stress_rate").empty()
                                            ? _X.name().with_suffix("_rate")
                                            : options.get<VariableName>("back_stress_rate"))),
    _C(declare_parameter<Scalar>("C", "C", true)),
    _g(declare_parameter<Scalar>("g", "g", true))
{
}

void
FredrickArmstrongPlasticHardening::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto eps = machine_precision(_X.scalar_type());
  // The effective stress
  auto s = neml2::norm(_X(), eps);
  // The part that's proportional to the plastic strain rate
  auto g_term = 2.0 / 3.0 * _C * _NM - _g * _X;

  if (out)
    _X_dot = g_term * _gamma_dot;

  if (dout_din)
  {
    auto I = imap<SR2>(_X.options()).template as<Tensor>();

    if (_gamma_dot.is_dependent())
      _X_dot.d(_gamma_dot) = g_term;

    if (_NM.is_dependent())
      _X_dot.d(_NM) = 2.0 / 3.0 * _C * _gamma_dot * I;

    if (_X.is_dependent())
      _X_dot.d(_X) = -_g * _gamma_dot * I;

    if (const auto * const C = nl_param("C"))
      _X_dot.d(*C) = 2.0 / 3.0 * _NM * _gamma_dot;

    if (const auto * const g = nl_param("g"))
      _X_dot.d(*g) = -_X * _gamma_dot;
  }
}

} // namespace neml2
