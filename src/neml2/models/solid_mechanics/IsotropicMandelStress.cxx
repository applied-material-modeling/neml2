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

#include "neml2/models/solid_mechanics/IsotropicMandelStress.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(IsotropicMandelStress);

OptionSet
IsotropicMandelStress::expected_options()
{
  OptionSet options = MandelStress::expected_options();
  options.doc() += " For isotropic material under small deformation, the Mandel stress and the "
                   "Cauchy stress coincide.";

  return options;
}

void
IsotropicMandelStress::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // Isotropic mandel stress is just the Cauchy stress

  if (out)
    _M = SR2(_S);

  if (dout_din)
    if (_S.is_dependent())
      _M.d(_S) = SR2::identity_map(_S.options());

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
