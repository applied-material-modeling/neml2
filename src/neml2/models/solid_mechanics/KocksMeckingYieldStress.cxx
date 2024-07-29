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

#include "neml2/models/solid_mechanics/KocksMeckingYieldStress.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(KocksMeckingYieldStress);

OptionSet
KocksMeckingYieldStress::expected_options()
{
  OptionSet options = NonlinearParameter<Scalar>::expected_options();

  options.doc() = "The yield stress given by the Kocks-Mecking model.  \\f$ \\sigma_y = \\exp{C} "
                  "\\mu \\f$ with \\f$ \\mu \\f$ the shear modulus and \\f$ C \\f$ the horizontal "
                  "intercept from the Kocks-Mecking diagram.";

  options.set_parameter<CrossRef<Scalar>>("C");
  options.set("C").doc() = "The Kocks-Mecking horizontal intercept";
  options.set_parameter<CrossRef<Scalar>>("shear_modulus");
  options.set("shear_modulus").doc() = "The shear modulus";

  return options;
}

KocksMeckingYieldStress::KocksMeckingYieldStress(const OptionSet & options)
  : NonlinearParameter<Scalar>(options),
    _C(declare_parameter<Scalar>("C", "C", /*allow_nonlinear=*/true)),
    _mu(declare_parameter<Scalar>("mu", "shear_modulus", /*allow_nonlinear=*/true))
{
}

void
KocksMeckingYieldStress::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _p = _mu * math::exp(_C);

  if (dout_din)
  {
    if (const auto * const mu = nl_param("mu"))
      _p.d(*mu) = math::exp(_C);

    if (const auto * const C = nl_param("C"))
      _p.d(*C) = _mu * math::exp(_C);
  }

  if (d2out_din2)
  {
    if (const auto * const C = nl_param("C"))
    {
      _p.d(*C, *C) = _mu * math::exp(_C);

      if (const auto * const mu = nl_param("mu"))
      {
        _p.d(*C, *mu) = math::exp(_C);
        _p.d(*mu, *C) = math::exp(_C);
      }
    }
  }
}
} // namespace neml2
