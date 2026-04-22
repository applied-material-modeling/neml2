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

#include "neml2/models/common/ForwardEulerTimeIntegration.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
template <typename T>
OptionSet
ForwardEulerTimeIntegration<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Perform forward Euler time integration defined as \\f$ s = s_n + (t - t_n) \\dot{s} "
      "\\f$, where \\f$s\\f$ is the variable being integrated, \\f$\\dot{s}\\f$ is the variable "
      "rate, and \\f$t\\f$ is time. Subscripts \\f$n\\f$ denote quantities from the previous time "
      "step.";

  options.add_output("variable", "Variable being integrated");
  options.add_input("time", "Time");

  return options;
}

template <typename T>
ForwardEulerTimeIntegration<T>::ForwardEulerTimeIntegration(const OptionSet & options)
  : Model(options),
    _s(declare_output_variable<T>("variable")),
    _sn(declare_input_variable<T>(history_name(_s.name(), /*nstep=*/1))),
    _rate(declare_input_variable<T>(rate_name(_s.name()))),
    _t(declare_input_variable<Scalar>("time")),
    _tn(declare_input_variable<Scalar>(history_name(_t.name(), /*nstep=*/1)))
{
}

template <typename T>
void
ForwardEulerTimeIntegration<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _s = _sn + _rate * (_t - _tn);

  if (dout_din)
  {
    auto I = imap_v<T>(_rate.options());

    _s.d(_rate) = I * (_t - _tn);
    _s.d(_sn) = I;
    _s.d(_t) = _rate();
    _s.d(_tn) = -_rate;
  }
}

#define REGISTER(T)                                                                                \
  using T##ForwardEulerTimeIntegration = ForwardEulerTimeIntegration<T>;                           \
  register_NEML2_object(T##ForwardEulerTimeIntegration);                                           \
  template class ForwardEulerTimeIntegration<T>
REGISTER(Scalar);
REGISTER(Vec);
REGISTER(SR2);
} // namespace neml2
