// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/models/solid_mechanics/YieldFunction.h"

namespace neml2
{
register_NEML2_object(YieldFunction);

OptionSet
YieldFunction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Classical macroscale plasticity yield function, \\f$ f = \\bar{\\sigma} - \\sigma_y - h "
      "\\f$, where \\f$ \\bar{\\sigma} \\f$ is the effective stress, \\f$ \\sigma_y \\f$ is the "
      "yield stress, and \\f$ h \\f$ is the isotropic hardening.";

  options.set<CrossRef<Scalar>>("yield_stress");
  options.set("yield_stress").doc() = "Yield stress";

  options.set<VariableName>("effective_stress") = VariableName("state", "internal", "s");
  options.set("effective_stress").doc() = "Effective stress";

  options.set<VariableName>("isotropic_hardening");
  options.set("isotropic_hardening").doc() = "Isotropic hardening";

  options.set<VariableName>("yield_function") = VariableName("state", "internal", "fp");
  options.set("yield_function").doc() = "Yield function";

  return options;
}

YieldFunction::YieldFunction(const OptionSet & options)
  : Model(options),
    _s(declare_input_variable<Scalar>("effective_stress")),
    _h(options.get<VariableName>("isotropic_hardening").empty()
           ? nullptr
           : &declare_input_variable<Scalar>("isotropic_hardening")),
    _f(declare_output_variable<Scalar>("yield_function")),
    _sy(declare_parameter<Scalar>("sy", "yield_stress"))
{
}

void
YieldFunction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
  {
    if (_h)
      _f = std::sqrt(2.0 / 3.0) * (_s - _sy - (*_h));
    else
      _f = std::sqrt(2.0 / 3.0) * (_s - _sy);
  }

  if (dout_din)
  {
    auto I = Scalar::identity_map(options());

    _f.d(_s) = std::sqrt(2.0 / 3.0) * I;

    if (_h)
      _f.d(*_h) = -std::sqrt(2.0 / 3.0) * I;

    if (const auto sy = nl_param("sy"))
      _f.d(*sy) = -std::sqrt(2.0 / 3.0) * I;
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
