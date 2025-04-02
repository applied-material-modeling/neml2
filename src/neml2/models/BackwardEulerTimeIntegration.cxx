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

#include "neml2/models/BackwardEulerTimeIntegration.h"

namespace neml2
{
template <typename T>
OptionSet
BackwardEulerTimeIntegration<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Define the backward Euler time integration residual \\f$ r = s - s_n - (t - t_n) \\dot{s} "
      "\\f$, where \\f$s\\f$ is the variable being integrated, \\f$\\dot{s}\\f$ is the variable "
      "rate, and \\f$t\\f$ is time. Subscripts \\f$n\\f$ denote quantities from the previous time "
      "step.";

  NonlinearSystem::enable_automatic_scaling(options);

  options.set_input("variable");
  options.set("variable").doc() = "Variable being integrated";

  options.set_input("rate");
  options.set("rate").doc() = "Variable rate";

  options.set_input("time") = VariableName(FORCES, "t");
  options.set("time").doc() = "Time";

  return options;
}

template <typename T>
BackwardEulerTimeIntegration<T>::BackwardEulerTimeIntegration(const OptionSet & options)
  : Model(options),
    _s(declare_input_variable<T>("variable")),
    _sn(declare_input_variable<T>(_s.name().old())),
    _ds_dt(options.get<VariableName>("rate").empty()
               ? declare_input_variable<T>(_s.name().with_suffix("_rate"))
               : declare_input_variable<T>("rate")),
    _t(declare_input_variable<Scalar>("time")),
    _tn(declare_input_variable<Scalar>(_t.name().old())),
    _r(declare_output_variable<T>(_s.name().remount(RESIDUAL)))
{
}

template <typename T>
void
BackwardEulerTimeIntegration<T>::diagnose() const
{
  Model::diagnose();
  diagnostic_assert_state(_s);
  diagnostic_assert_state(_ds_dt);
  diagnostic_assert_force(_t);
}

template <typename T>
void
BackwardEulerTimeIntegration<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _r = _s - _sn - _ds_dt * (_t - _tn);

  if (dout_din)
  {
    auto I = T::identity_map(_s.options());

    _r.d(_s) = I;
    _r.d(_ds_dt) = -I * (_t - _tn);

    if (currently_solving_nonlinear_system())
      return;

    _r.d(_sn) = -I;
    _r.d(_t) = -_ds_dt;
    _r.d(_tn) = _ds_dt;
  }
}

#define REGISTER(T)                                                                                \
  using T##BackwardEulerTimeIntegration = BackwardEulerTimeIntegration<T>;                         \
  register_NEML2_object(T##BackwardEulerTimeIntegration);                                          \
  template class BackwardEulerTimeIntegration<T>
REGISTER(Scalar);
REGISTER(Vec);
REGISTER(SR2);
REGISTER(R2);
} // namespace neml2
