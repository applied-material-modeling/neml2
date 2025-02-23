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

#include "neml2/models/solid_mechanics/LinearIsotropicHardening.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(LinearIsotropicHardening);

OptionSet
LinearIsotropicHardening::expected_options()
{
  OptionSet options = IsotropicHardening::expected_options();
  options.doc() += " following a linear relationship, i.e., \\f$ h = K \\varepsilon_p \\f$ where "
                   "\\f$ K \\f$ is the hardening modulus.";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("hardening_modulus");
  options.set("hardening_modulus").doc() = "Hardening modulus";

  return options;
}

LinearIsotropicHardening::LinearIsotropicHardening(const OptionSet & options)
  : IsotropicHardening(options),
    _K(declare_parameter<Scalar>("K", "hardening_modulus"))
{
}

void
LinearIsotropicHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _h = _K * _ep;

  if (dout_din)
    if (_ep.is_dependent())
      _h.d(_ep) = _K;

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
