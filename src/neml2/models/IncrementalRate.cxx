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

#include "neml2/models/IncrementalRate.h"

namespace neml2
{
register_NEML2_object(R2IncrementalRate);

template <typename T>
OptionSet
IncrementalRate<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the first order discrete time derivative of a variable as \\f$ "
                  "\\dot{f} = \\frac{\\Delta f}{t-t_n} \\f$, where \\f$ \\Deltaf \\f$ is the "
                  "incremental force variable, "
                  "and \\f$ t \\f$ is time.";

  options.set_output("rate");
  options.set("rate").doc() = "The variable's rate of change";

  options.set_input("variable");
  options.set("variable").doc() = "The variable to take time derivative with";

  options.set_input("time") = VariableName(FORCES, "t");
  options.set("time").doc() = "Time";

  return options;
}

template <typename T>
IncrementalRate<T>::IncrementalRate(const OptionSet & options)
  : Model(options),
    _dv(declare_input_variable<T>("variable")),
    _t(declare_input_variable<Scalar>("time")),
    _tn(declare_input_variable<Scalar>(_t.name().old())),
    _dv_dt(options.get<VariableName>("rate").empty()
               ? declare_output_variable<T>(_dv.name().with_suffix("_rate"))
               : declare_output_variable<T>("rate"))
{
}

template <typename T>
void
IncrementalRate<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert(!d2out_din2, "IncrementalRate does not implement second derivatives");

  auto dt = _t - _tn;

  if (out)
    _dv_dt = _dv / dt;

  if (dout_din)
  {
    if (_dv.is_dependent())
      _dv_dt.d(_dv) = T::identity_map(_dv.options()) / dt;
    if (_t.is_dependent())
      _dv_dt.d(_t) = -_dv / dt / dt;
    if (_tn.is_dependent())
      _dv_dt.d(_tn) = _dv / dt / dt;
  }
}

template class IncrementalRate<R2>;
} // namespace neml2
