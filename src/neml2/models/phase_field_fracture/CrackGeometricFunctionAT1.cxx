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

#include "neml2/models/phase_field_fracture/CrackGeometricFunctionAT1.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(CrackGeometricFunctionAT1);

OptionSet
CrackGeometricFunctionAT1::expected_options()
{
  OptionSet options = CrackGeometricFunction::expected_options();
  options.doc() =
      "Crack geometric function associated with the AT-1 functional, $ \\alpha = d $";
  options.set<bool>("define_second_derivatives") = true;

  return options;
}

CrackGeometricFunctionAT1::CrackGeometricFunctionAT1(const OptionSet & options)
  : CrackGeometricFunction(options)
{
}

void
CrackGeometricFunctionAT1::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
  {
    _alpha = _d;
  }

  if (dout_din)
  {
    _alpha.d(_d) = Scalar::create(1.0, _d.options());
  }

  if (d2out_din2)
  {
    _alpha.d(_d, _d) = Scalar::create(0.0, _d.options());
  }
}
} // namespace neml2
