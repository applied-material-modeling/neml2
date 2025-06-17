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

#include "neml2/models/porous_flow/KozenyCarmanPermeability.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(KozenyCarmanPermeability);
OptionSet
KozenyCarmanPermeability::expected_options()
{
  OptionSet options = PorosityPermeabilityRelation::expected_options();
  options.doc() +=
      " The Kozeny-Carman porosity-permeability relation takes the form of \\f$ K = K_0 "
      "\\frac{\\varphi^n (1-\\varphi_0^m)}{\\varphi_0^m (1-\\varphi)^n} \\f$ where "
      "\\f$ n \\f$ and \\f$ m \\f$ are shape parameters. \\f$ varphi_0 \\f$ and \\f$ "
      "K_0 \\f$ are the reference porosity and permeability respectively.";

  options.set_parameter<TensorName<Scalar>>("reference_permeability");
  options.set("reference_permeability").doc() = "the reference permeability";

  options.set_parameter<TensorName<Scalar>>("reference_porosity");
  options.set("reference_porosity").doc() = "the reference porosity";

  options.set_parameter<TensorName<Scalar>>("n");
  options.set("n").doc() = "Shape parameter n";

  options.set_parameter<TensorName<Scalar>>("m");
  options.set("m").doc() = "Shape parameter m";

  return options;
}

KozenyCarmanPermeability::KozenyCarmanPermeability(const OptionSet & options)
  : PorosityPermeabilityRelation(options),
    _K0(declare_parameter<Scalar>("K0", "reference_permeability")),
    _phi0(declare_parameter<Scalar>("phi0", "reference_porosity")),
    _n(declare_parameter<Scalar>("n", "n")),
    _m(declare_parameter<Scalar>("m", "m"))
{
}

void
KozenyCarmanPermeability::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _K = _K0 * pow(_phi / (1 - _phi), _n) / pow(_phi0 / (1 - _phi0), _m);

  if (dout_din)
    _K.d(_phi) = _K0 / pow(_phi0 / (1 - _phi0), _m) * (_n * pow(_phi / (1 - _phi), _n - 1)) /
                 (1 - _phi) / (1 - _phi);
}
}
