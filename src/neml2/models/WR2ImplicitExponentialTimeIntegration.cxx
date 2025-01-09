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

#include "neml2/models/WR2ImplicitExponentialTimeIntegration.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
register_NEML2_object(WR2ImplicitExponentialTimeIntegration);

OptionSet
WR2ImplicitExponentialTimeIntegration::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Define the implicit discrete exponential time integration residual of a "
                  "rotation variable. The residual can be written as \\f$ r = s - \\exp\\left[ "
                  "(t-t_n)\\dot{s}\\right] \\circ s_n \\f$, where \\f$ \\circ \\f$ denotes the "
                  "rotation operator.";

  NonlinearSystem::enable_automatic_scaling(options);

  options.set_input("variable");
  options.set("variable").doc() = "Variable being integrated";

  options.set_input("rate");
  options.set("rate").doc() = "Variable rate";

  options.set_input("time") = VariableName(FORCES, "t");
  options.set("time").doc() = "Time";

  return options;
}

WR2ImplicitExponentialTimeIntegration::WR2ImplicitExponentialTimeIntegration(
    const OptionSet & options)
  : Model(options),
    _s(declare_input_variable<Rot>("variable")),
    _sn(declare_input_variable<Rot>(_s.name().old())),
    _s_dot(options.get<VariableName>("rate").empty()
               ? declare_input_variable<WR2>(_s.name().with_suffix("_rate"))
               : declare_input_variable<WR2>("rate")),
    _t(declare_input_variable<Scalar>("time")),
    _tn(declare_input_variable<Scalar>(_t.name().old())),
    _r(declare_output_variable<Rot>(_s.name().remount(RESIDUAL)))
{
}

void
WR2ImplicitExponentialTimeIntegration::diagnose() const
{
  Model::diagnose();
  diagnostic_assert_state(_s);
  diagnostic_assert_state(_s_dot);
  diagnostic_assert_force(_t);
}

void
WR2ImplicitExponentialTimeIntegration::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto dt = _t - _tn;
  const auto inc = (_s_dot * dt).exp();

  if (out)
    _r = _s - Rot(_sn).rotate(inc);

  if (dout_din)
  {
    const auto de = (_s_dot * dt).dexp();
    _r.d(_s) = R2::identity(_s.options());
    _r.d(_s_dot) = -Rot(_sn).drotate(inc) * de * dt;

    if (currently_solving_nonlinear_system())
      return;

    _r.d(_sn) = -Rot(_sn).drotate_self(inc);
    _r.d(_t) = -Rot(_sn).drotate(inc) * de * Vec(_s_dot.value());
    _r.d(_tn) = Rot(_sn).drotate(inc) * de * Vec(_s_dot.value());
  }
}
} // namespace neml2
