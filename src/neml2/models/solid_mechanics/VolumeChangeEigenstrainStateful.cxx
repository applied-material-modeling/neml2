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

#include "neml2/models/solid_mechanics/VolumeChangeEigenstrainStateful.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(VolumeChangeEigenstrainStateful);

OptionSet
VolumeChangeEigenstrainStateful::expected_options()
{
  OptionSet options = Eigenstrain::expected_options();
  options.doc() =
      "Define the (cummulative, as opposed to instantaneous) linear isotropic volume expansion "
      "eigenstrain, "
      "i.e. \\f$ \\boldsymbol{\\varepsilon}_V = \\(V/V0)^(1/3)-1 \\boldsymbol{I} \\f$, where "
      " \\f$ V \\f$ is the current volume, "
      "and \\f$ V0 \\f$ is the reference (initial) volume, which is a state parameter.";

  options.set_input("volume") = VariableName(STATE, "V");
  options.set("volume").doc() = "Volume";

  options.set_input("reference_volume") = VariableName(STATE, "V0");
  options.set("reference_volume").doc() = "Reference volume";

  return options;
}

VolumeChangeEigenstrainStateful::VolumeChangeEigenstrainStateful(const OptionSet & options)
  : Eigenstrain(options),
    _V0(declare_input_variable<Scalar>("reference_volume")),
    _V(declare_input_variable<Scalar>("volume"))
{
}

void
VolumeChangeEigenstrainStateful::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _eg = (pow(_V / _V0, (1.0 / 3.0)) - 1.0) * SR2::identity(_V.options());

  if (dout_din)
  {
    if (_V.is_dependent())
    {
      _eg.d(_V) = 1.0 / (3 * _V0) * pow(_V / _V0, (-2.0 / 3.0)) * SR2::identity(_V.options());
      _eg.d(_V0) = -1.0 / (3 * _V0) * pow(_V / _V0, (1.0 / 3.0)) * SR2::identity(_V.options());
    }
  }

  if (d2out_din2)
  {
    if (_V.is_dependent())
      _eg.d(_V, _V) =
          -2.0 / (9.0 * _V0 * _V0) * pow(_V / _V0, (-5.0 / 3.0)) * SR2::identity(_V.options());
  }
}
}
// namespace neml2
