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

#include "neml2/models/pyrolysis/ThermalDecompositionConversionDegreeConversionDegree.h"

namespace neml2
{
register_NEML2_object(ThermalDecompositionConversionDegreeConversionDegree);

OptionSet
ThermalDecompositionConversionDegreeConversionDegree::expected_options()
{
  OptionSet options = ConversionDegree::expected_options();
  options.doc() =
      "Calculate the conversion degree associated with a thermal decomposition process where a "
      "precursor solid A decomposes into the residue solid B plus byproducts. The conversion "
      "degree is defined by \f$ \\dfrac{Y}{1-Y} \\dfrac{w_p}{w_p^0} + 1 \f$, where \f$ Y \f$ is "
      "the experimentally measured yield (between 0 and 1) of the decomposition denoting the mass "
      "ratio between final residue and initial precursor, and \f$ w_p^0 \f$ is the initial mass "
      "fraction of the precursor";

  options.set_parameter<TensorName<Scalar>>("initial_precursor_mass_fraction");
  options.set("initial_precursor_mass_fraction").doc() =
      "The precursor's initial mass fraction before decomposition";

  options.set_parameter<TensorName<Scalar>>("reaction_yield");
  options.set("reaction_yield").doc() = "The final reaction yield (between 0 and 1)";

  options.set_input("precursor_mass_fraction") = VariableName("state", "wp");
  options.set("precursor_mass_fraction").doc() = "The precursor's mass fraction.";

  return options;
}

ThermalDecompositionConversionDegreeConversionDegree::
    ThermalDecompositionConversionDegreeConversionDegree(const OptionSet & options)
  : ConversionDegree(options),
    _wp0(declare_parameter<Scalar>("wp0", "initial_precursor_mass_fraction")),
    _Y(declare_parameter<Scalar>("Y", "reaction_yield")),
    _wp(declare_input_variable<Scalar>("precursor_mass_fraction"))
{
}

void
ThermalDecompositionConversionDegreeConversionDegree::set_value(bool out,
                                                                bool dout_din,
                                                                bool /*d2out_din2*/)
{
  auto coef = _Y / (1 - _Y) / _wp0;

  if (out)
    _a = coef * _wp + 1.0;

  if (dout_din)
    _a.d(_wp) = coef;
}
}
