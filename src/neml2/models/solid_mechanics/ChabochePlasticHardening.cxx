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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/log.h"
#include "neml2/tensors/functions/norm.h"
#include "neml2/tensors/functions/outer.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(ChabochePlasticHardening);

OptionSet
ChabochePlasticHardening::expected_options()
{
  OptionSet options = FredrickArmstrongPlasticHardening::expected_options();
  options.doc() +=
      " The complete Chaboche model adds static recovery terms \\f$ - A \\lVert \\boldsymbol{X} "
      "\\rVert^{a - 1} \\boldsymbol{X} \\f$, so the model includes kinematic hardening, dynamic "
      "recovery, and static recovery.  \\f$ A \\f$ and \\f$ a \\f$ are additional material "
      "parameters.";

  options.set_parameter<TensorName<Scalar>>("A");
  options.set("A").doc() = "Static recovery prefactor";

  options.set_parameter<TensorName<Scalar>>("a");
  options.set("a").doc() = "Static recovery exponent";

  return options;
}

ChabochePlasticHardening::ChabochePlasticHardening(const OptionSet & options)
  : FredrickArmstrongPlasticHardening(options),
    _A(declare_parameter<Scalar>("A", "A", true)),
    _a(declare_parameter<Scalar>("a", "a", true))
{
}

void
ChabochePlasticHardening::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto eps = machine_precision(_X.scalar_type());
  // The effective stress
  auto s = neml2::norm(_X(), eps);
  // The part that's proportional to the plastic strain rate
  auto g_term = 2.0 / 3.0 * _C * _NM - _g * _X;
  // The static recovery term
  auto s_term = -_A * pow(s, _a - 1) * _X;

  if (out)
    _X_dot = g_term * _gamma_dot + s_term;

  if (dout_din)
  {
    auto I = imap<SR2>(_X.options()).as<SSR4>();

    if (_gamma_dot.is_dependent())
      _X_dot.d(_gamma_dot) = g_term;

    if (_NM.is_dependent())
      _X_dot.d(_NM) = 2.0 / 3.0 * _C * _gamma_dot * I;

    if (_X.is_dependent())
      _X_dot.d(_X) =
          -_g * _gamma_dot * I - _A * pow(s, _a - 3) * ((_a - 1) * neml2::outer(_X()) + s * s * I);

    if (const auto * const C = nl_param("C"))
      _X_dot.d(*C) = 2.0 / 3.0 * _NM * _gamma_dot;

    if (const auto * const g = nl_param("g"))
      _X_dot.d(*g) = -_X * _gamma_dot;

    if (const auto * const A = nl_param("A"))
      _X_dot.d(*A) = -pow(s, _a - 1) * _X;

    if (const auto * const a = nl_param("a"))
      _X_dot.d(*a) = -_A * _X * pow(s, _a - 1) * log(s);
  }
}

} // namespace neml2
