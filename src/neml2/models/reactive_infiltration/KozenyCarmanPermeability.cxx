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

#include "neml2/models/reactive_infiltration/KozenyCarmanPermeability.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(KozenyCarmanPermeability);
OptionSet
KozenyCarmanPermeability::expected_options()
{
  OptionSet options = PorosityPermeabilityRelation::expected_options();
  options.doc() =
      "Define the porous Kozeny Carman porosity-permeability relation, takes the form of "
      "\\f$ K_o \\frac{\\varphi^n(1-\\varphi_o^m)}{\\varphi_o^m(1-\\varphi)^n} \\f$ where \\f$ n, "
      "m "
      "\\f$ is the fitting parameter. \\f$ varphi_o, K_o \\f$ are the reference porosity and "
      "permeability respectively.";

  options.set_parameter<TensorName<Scalar>>("power");
  options.set("power").doc() = "value of power constant n, associated with state variables";

  options.set_parameter<TensorName<Scalar>>("reference_power");
  options.set("reference_power").doc() =
      "value of power constant m, associated with reference porosity";

  return options;
}

KozenyCarmanPermeability::KozenyCarmanPermeability(const OptionSet & options)
  : PorosityPermeabilityRelation(options),
    _n(declare_parameter<Scalar>("n", "power")),
    _m(declare_parameter<Scalar>("m", "reference_power"))
{
}

void
KozenyCarmanPermeability::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _K = _Ko * (pow(_phi, _n) * pow(1 - _phio, _m)) / (pow(_phio, _m) * pow(1 - _phi, _n));
  }

  if (dout_din)
  {
    _K.d(_phi) = _Ko * pow(1 - _phio, _m) / pow(_phio, _m) * (_n * pow((_phi) / (1 - _phi), _n)) /
                 (_phi - _phi * _phi);
  }
}
}