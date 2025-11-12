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

#include "neml2/models/solid_mechanics/RateIndependentPlasticFlowConstraint.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(RateIndependentPlasticFlowConstraint);

OptionSet
RateIndependentPlasticFlowConstraint::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Solve the consistent plasticity yield envelope by solving the equivalent "
                  "complementarity condition \\f[ r = \\dot{\\gamma} - f^p - "
                  "\\sqrt{{\\dot{\\gamma}}^2 + {f^p}^2} \\f]";

  options.set_input("yield_function") = VariableName(STATE, "internal", "fp");
  options.set("yield_function").doc() = "Yield function";

  options.set_input("flow_rate") = VariableName(STATE, "internal", "gamma_rate");
  options.set("flow_rate").doc() = "Flow rate";

  return options;
}

RateIndependentPlasticFlowConstraint::RateIndependentPlasticFlowConstraint(
    const OptionSet & options)
  : Model(options),
    _fp(declare_input_variable<Scalar>("yield_function")),
    _gamma_dot(declare_input_variable<Scalar>("flow_rate")),
    _r(declare_output_variable<Scalar>(_gamma_dot.name().remount(RESIDUAL)))
{
}

void
RateIndependentPlasticFlowConstraint::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto FB = _gamma_dot - _fp - sqrt(_gamma_dot * _gamma_dot + _fp * _fp);

  if (out)
    _r = FB;

  if (dout_din)
  {
    const auto I = imap_v<Scalar>(_gamma_dot.options());
    const auto eps = machine_precision(_gamma_dot.scalar_type());

    if (_gamma_dot.is_dependent())
      _r.d(_gamma_dot) = I - _gamma_dot / sqrt(_gamma_dot * _gamma_dot + _fp * _fp + eps);

    if (_fp.is_dependent())
      _r.d(_fp) = -I - _fp / sqrt(_gamma_dot * _gamma_dot + _fp * _fp + eps);
  }
}

} // namespace neml2
