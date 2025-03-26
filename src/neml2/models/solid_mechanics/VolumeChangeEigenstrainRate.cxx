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

#include "neml2/models/solid_mechanics/VolumeChangeEigenstrainRate.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(VolumeChangeEigenstrainRate);

OptionSet
VolumeChangeEigenstrainRate::expected_options()
{
  OptionSet options = EigenstrainRate::expected_options();
  options.doc() = "Define the volumetric eigenstrain rate, takes the form \\f$ "
                  "\\dot{\\varepsilon}_v = \\frac{\\dot{V}}{3V}(\\varepsilon_v+1) \\f$. where \\f$ "
                  "\\varepsilon_v \\f$ is the eigenstrain and \\f$ V \\f$ is the volume";

  options.set_input("volume") = VariableName(STATE, "volume");
  options.set("volume").doc() = "Volume";

  options.set_input("volume_rate") = VariableName(STATE, "volume_rate");
  options.set("volume_rate").doc() = "Time rate of change of volume";

  options.set_input("eigenstrain") = VariableName(STATE, "eigenstrain");
  options.set("eigenstrain").doc() = "The eigenstrain.";

  return options;
}

VolumeChangeEigenstrainRate::VolumeChangeEigenstrainRate(const OptionSet & options)
  : EigenstrainRate(options),
    _V(declare_input_variable<Scalar>("volume")),
    _Vdot(declare_input_variable<Scalar>("volume_rate")),
    _eg(declare_input_variable<SR2>("eigenstrain"))
{
}

void
VolumeChangeEigenstrainRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto I = SR2::identity(_eg.options());

  auto egdot = (_eg + I) * (1.0 / 3.0 * _Vdot / _V);

  if (out)
    _egdot = egdot;

  if (dout_din)
  {
    if (_V.is_dependent())
    {
      _egdot.d(_V) = -egdot / _V;
      _egdot.d(_Vdot) = (_eg + I) * (1.0 / 3.0 / _V);
    }
    _egdot.d(_eg) = 1.0 / 3.0 * _Vdot / _V * SR2::identity_map(_eg.options());
  }
}
}
// namespace neml2
