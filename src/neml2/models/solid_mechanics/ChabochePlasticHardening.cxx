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

#include "neml2/models/solid_mechanics/ChabochePlasticHardening.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ChabochePlasticHardening);

OptionSet
ChabochePlasticHardening::expected_options()
{
  OptionSet options = FlowRule::expected_options();
  options.doc() +=
      " This object defines the non-associative Chaboche kinematic hardening. In the Chaboche "
      "model, back stress is directly treated as an internal variable. Rate of back stress is "
      "given as \\f$ \\dot{\\boldsymbol{X}} = \\left( \\frac{2}{3} C \\frac{\\partial f}{\\partial "
      "\\boldsymbol{M}} - g \\boldsymbol{X} \\right) \\dot{\\gamma} - A \\lVert \\boldsymbol{X} "
      "\\rVert^{a - 1} \\boldsymbol{X} \\f$, including kinematic hardening, dynamic recovery, and "
      "static recovery. \\f$ \\frac{\\partial f}{\\partial \\boldsymbol{M}} \\f$ is the flow "
      "direction, \\f$ \\dot{\\gamma} \\f$ is the flow rate, and \\f$ C \\f$, \\f$ g \\f$, \\f$ A "
      "\\f$, and \\f$ a \\f$ are material parameters.";

  options.set_input("back_stress") = VariableName("state", "internal", "X");
  options.set("back_stress").doc() = "Back stress";

  options.set_input("flow_direction") = VariableName("state", "internal", "NM");
  options.set("flow_direction").doc() = "Flow direction";

  options.set_parameter<CrossRef<Scalar>>("C");
  options.set("C").doc() = "Kinematic hardening coefficient";

  options.set_parameter<CrossRef<Scalar>>("g");
  options.set("g").doc() = "Dynamic recovery coefficient";

  options.set_parameter<CrossRef<Scalar>>("A");
  options.set("A").doc() = "Static recovery prefactor";

  options.set_parameter<CrossRef<Scalar>>("a");
  options.set("a").doc() = "Static recovery exponent";

  return options;
}

ChabochePlasticHardening::ChabochePlasticHardening(const OptionSet & options)
  : FlowRule(options),
    _X(declare_input_variable<SR2>("back_stress")),
    _NM(declare_input_variable<SR2>("flow_direction")),
    _X_dot(declare_output_variable<SR2>(_X.name().with_suffix("_rate"))),
    _C(declare_parameter<Scalar>("C", "C")),
    _g(declare_parameter<Scalar>("g", "g")),
    _A(declare_parameter<Scalar>("A", "A")),
    _a(declare_parameter<Scalar>("a", "a"))
{
}

void
ChabochePlasticHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2,
                  "ChabochePlasticHardening model doesn't implement second derivatives.");

  // The effective stress
  auto s = SR2(_X).norm(machine_precision());
  // The part that's proportional to the plastic strain rate
  auto g_term = 2.0 / 3.0 * _C * _NM - _g * _X;

  if (out)
  {
    // The static recovery term
    auto s_term = -_A * math::pow(s, _a - 1) * _X;
    _X_dot = g_term * _gamma_dot + s_term;
  }

  if (dout_din)
  {
    auto I = SR2::identity_map(options());

    if (_gamma_dot.is_dependent())
      _X_dot.d(_gamma_dot) = g_term;

    if (_NM.is_dependent())
      _X_dot.d(_NM) = 2.0 / 3.0 * _C * _gamma_dot * I;

    if (_X.is_dependent())
      _X_dot.d(_X) = -_g * _gamma_dot * I -
                     _A * math::pow(s, _a - 3) * ((_a - 1) * SR2(_X).outer(SR2(_X)) + s * s * I);
  }
}

} // namespace neml2
