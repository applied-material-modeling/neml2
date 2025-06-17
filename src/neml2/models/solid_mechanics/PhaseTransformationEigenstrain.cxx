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

#include "neml2/models/solid_mechanics/PhaseTransformationEigenstrain.h"

namespace neml2
{
register_NEML2_object(PhaseTransformationEigenstrain);

OptionSet
PhaseTransformationEigenstrain::expected_options()
{
  OptionSet options = Eigenstrain::expected_options();
  options.doc() =
      "Define the (cummulative, as opposed to instantaneous) linear isotropic phase transformation "
      "(from phase A to phase B) eigenstrain, i.e. \\f$ \\boldsymbol{\\varepsilon}_\\mathrm{PT} = "
      "\\Delta V f \\boldsymbol{I} \\f$, where \\f$ \\Delta V \\f$ is the volume fraction change "
      "when going from phase A to B, \\f$ f \\f$ is the phase fraction (0 to 1, A to B).";

  options.set_input("volume_fraction_change") = VariableName(STATE, "dv");
  options.set("volume_fraction_change").doc() =
      "Change in volume fraction going from phase A to phase B";

  options.set_input("phase_fraction") = VariableName(STATE, "f");
  options.set("phase_fraction").doc() = "Phase fraction";

  return options;
}

PhaseTransformationEigenstrain::PhaseTransformationEigenstrain(const OptionSet & options)
  : Eigenstrain(options),
    _f(declare_input_variable<Scalar>("phase_fraction")),
    _dv(declare_input_variable<Scalar>("volume_fraction_change"))
{
}

void
PhaseTransformationEigenstrain::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _eg = _f * _dv * SR2::identity(_f.options());

  if (dout_din)
  {
    _eg.d(_f) = _dv * SR2::identity(_f.options());
    _eg.d(_dv) = _f * SR2::identity(_f.options());
  }

  if (d2out_din2)
  {
    _eg.d(_f, _dv) = SR2::identity(_f.options());
    _eg.d(_dv, _f) = SR2::identity(_f.options());
  }
}
}
// namespace neml2
