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

#include "neml2/models/IsotropicSR2.h"

namespace neml2
{
register_NEML2_object(IsotropicSR2);

OptionSet
IsotropicSR2::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Scale a symmetric isotropic second order tensor by a factor";

  options.set_output("output");
  options.set("output").doc() = "Output variable";

  options.set_parameter<TensorName<Scalar>>("factor");
  options.set("factor").doc() = "Scale factor";

  return options;
}

IsotropicSR2::IsotropicSR2(const OptionSet & options)
  : Model(options),
    _s(declare_parameter<Scalar>("f", "factor", /*allow_nonlinear=*/true)),
    _y(declare_output_variable<SR2>("output"))
{
}

void
IsotropicSR2::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _y = _s * SR2::identity(_s.options());

  if (dout_din)
    if (const auto * const s = nl_param("f"))
      _y.d(*s) = SR2::identity(_s.options());
}
} // namespace neml2
