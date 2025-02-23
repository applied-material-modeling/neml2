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

#include "neml2/models/solid_mechanics/LinearKinematicHardening.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(LinearKinematicHardening);

OptionSet
LinearKinematicHardening::expected_options()
{
  OptionSet options = KinematicHardening::expected_options();
  options.doc() += " following a linear relationship, i.e., \\f$ \\boldsymbol{X} = H "
                   "\\boldsymbol{K}_p \\f$ where \\f$ H \\f$ is the hardening modulus.";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("hardening_modulus");
  options.set("hardening_modulus").doc() = "Hardening modulus";

  return options;
}

LinearKinematicHardening::LinearKinematicHardening(const OptionSet & options)
  : KinematicHardening(options),
    _H(declare_parameter<Scalar>("H", "hardening_modulus"))
{
}

void
LinearKinematicHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _X = _H * _Kp;

  if (dout_din)
    if (_Kp.is_dependent())
      _X.d(_Kp) = _H * SR2::identity_map(_H.options());

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
