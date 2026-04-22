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

#include "neml2/models/common/WR2ExplicitExponentialTimeIntegration.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
register_NEML2_object(WR2ExplicitExponentialTimeIntegration);

OptionSet
WR2ExplicitExponentialTimeIntegration::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Perform explicit discrete exponential time integration of a rotation. The "
                  "update can be written as \\f$ s = \\exp\\left[ (t-t_n)\\dot{s}\\right] \\circ "
                  "s_n \\f$, where \\f$ \\circ \\f$ denotes the rotation operator.";

  options.add_output("variable", "Variable being integrated");
  options.add_input("time", "Time");

  return options;
}

WR2ExplicitExponentialTimeIntegration::WR2ExplicitExponentialTimeIntegration(
    const OptionSet & options)
  : Model(options),
    _s(declare_output_variable<Rot>("variable")),
    _sn(declare_input_variable<Rot>(history_name(_s.name(), /*nstep=*/1))),
    _rate(declare_input_variable<WR2>(rate_name(_s.name()))),
    _t(declare_input_variable<Scalar>("time")),
    _tn(declare_input_variable<Scalar>(history_name(_t.name(), /*nstep=*/1)))
{
}

void
WR2ExplicitExponentialTimeIntegration::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto dt = _t - _tn;

  // Incremental rotation
  const auto inc = (_rate * dt).exp_map();

  if (out)
    _s = _sn().rotate(inc);

  if (dout_din)
  {
    const auto de = (_rate * dt).dexp_map();
    _s.d(_rate) = _sn().drotate(inc) * de * dt;
    _s.d(_sn) = _sn().drotate_self(inc);
    _s.d(_t) = _sn().drotate(inc) * de * Vec(_rate());
    _s.d(_tn) = -_s.d(_t).tensor();
  }
}

} // namespace neml2
