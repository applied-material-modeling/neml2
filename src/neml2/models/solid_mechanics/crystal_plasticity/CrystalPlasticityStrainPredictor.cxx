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
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/functions/norm.h"

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
  options.add_buffer<Scalar>("scale", "Scale factor applied to the strain increment");
  options.add_buffer<Scalar>(
      "threshold",
      TensorName<Scalar>("1e-3"),
      "Only apply the predictor if the old elastic strain norm is below this threshold. When the "
      "old elastic strain norm is above this threshold, fall back to use the old elastic strain as "
      "the initial guess for the nonlinear solve.");

  return options;
}

CrystalPlasticityStrainPredictor::CrystalPlasticityStrainPredictor(const OptionSet & options)
  : Predictor(options),
    _D(declare_input_variable<SR2>("deformation_rate")),
    _t(declare_input_variable<Scalar>("time")),
    _tn(declare_input_variable<Scalar>(history_name(_t.name(), 1))),
    _scale(declare_buffer<Scalar>("scale", "scale")),
    _Ee(declare_output_variable<SR2>("elastic_strain")),
    _Ee_n(declare_input_variable<SR2>(history_name(_Ee.name(), 1))),
    _threshold(declare_buffer<Scalar>("threshold", "threshold"))
{
}

void
CrystalPlasticityStrainPredictor::predict()
{
  auto Ee_np1 = _Ee_n + _scale * _D * (_t - _tn);
  auto Ee_n = _Ee_n().batch_expand_as(Ee_np1);
  _Ee = neml2::where(neml2::norm(Ee_n) < _threshold, Ee_np1, Ee_n);
}
} // namespace neml2
