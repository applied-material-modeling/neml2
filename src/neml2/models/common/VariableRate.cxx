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

#include "neml2/models/common/VariableRate.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
template <typename T>
OptionSet
VariableRate<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the first order discrete time derivative of a variable as \\f$ "
      "\\dot{f} = \\frac{f-f_n}{t-t_n} \\f$, where \\f$ f \\f$ is the variable, "
      "\\f$ f_n \\f$ is the variable at the previous time step, and \\f$ t \\f$ is time.";

  options.add_input("variable", "The variable being differentiated");
  options.add_input("time", "t", "Time");

  return options;
}

template <typename T>
VariableRate<T>::VariableRate(const OptionSet & options)
  : Model(options),
    _v(declare_input_variable<T>("variable")),
    _vn(declare_input_variable<T>(history_name(_v.name(), /*nstep=*/1))),
    _t(declare_input_variable<Scalar>("time")),
    _tn(declare_input_variable<Scalar>(history_name(_t.name(), /*nstep=*/1))),
    _rate(declare_output_variable<T>(rate_name(_v.name())))
{
}

template <typename T>
void
VariableRate<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto dt = _t - _tn;
  auto dv = _v - _vn;
  auto rate = dv / dt;

  if (out)
    _rate = rate;

  if (dout_din)
  {
    _rate.d(_v) = imap_v<T>(_v.options()) / dt;
    _rate.d(_vn) = -imap_v<T>(_v.options()) / dt;
    _rate.d(_t) = -rate / dt;
    _rate.d(_tn) = rate / dt;
  }
}

#define REGISTER(T)                                                                                \
  using T##VariableRate = VariableRate<T>;                                                         \
  register_NEML2_object(T##VariableRate);                                                          \
  template class VariableRate<T>
REGISTER(Scalar);
REGISTER(Vec);
REGISTER(SR2);
REGISTER(R2);
} // namespace neml2
