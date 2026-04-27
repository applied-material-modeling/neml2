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

#include "neml2/models/solid_mechanics/crystal_plasticity/CrystalPlasticityStrainPredictor.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(CrystalPlasticityStrainPredictor);

OptionSet
CrystalPlasticityStrainPredictor::expected_options()
{
  OptionSet options = Predictor::expected_options();
  options.doc() =
      "Warm-up predictor for crystal plasticity models. Computes an initial guess for the elastic "
      "strain as \\f$ \\varepsilon^e = s \\cdot \\Delta t \\cdot d \\f$ where \\f$ \\Delta t = t "
      "- t_n \\f$ is the time increment, \\f$ d \\f$ is the deformation rate, and \\f$ s \\f$ is "
      "a scale factor.";

  options.add_input("deformation_rate", "Deformation rate tensor");
  options.add_input("time", "t", "Current time");
  options.add_output("elastic_strain", "Elastic strain initial guess");
  options.add_parameter<Scalar>("scale", "Scale factor applied to the strain increment");

  return options;
}

CrystalPlasticityStrainPredictor::CrystalPlasticityStrainPredictor(const OptionSet & options)
  : Predictor(options),
    _D(declare_input_variable<SR2>("deformation_rate")),
    _t(declare_input_variable<Scalar>("time")),
    _tn(declare_input_variable<Scalar>(history_name(_t.name(), 1))),
    _scale(declare_parameter<Scalar>("scale", "scale")),
    _Ee(declare_output_variable<SR2>("elastic_strain"))
{
}

void
CrystalPlasticityStrainPredictor::predict()
{
  _Ee = _scale * _D * (_t - _tn);
}
} // namespace neml2
