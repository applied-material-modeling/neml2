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

#include "neml2/models/solid_mechanics/elasticity/CubicElasticityTensor.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(CubicElasticityTensor);

OptionSet
CubicElasticityTensor::expected_options()
{
  OptionSet options = ElasticityInterface<NonlinearParameter<SSR4>, 3>::expected_options();
  options.doc() = "This class defines a cubic anisotropic elasticity tensor using three parameters."
                  "  Various options are available for which three parameters to provide.";

  return options;
}

CubicElasticityTensor::CubicElasticityTensor(const OptionSet & options)
  : ElasticityInterface<NonlinearParameter<SSR4>, 3>(options),
    _converter(_constant_types, _need_derivs)
{
}

void
CubicElasticityTensor::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto [C1_and_dC1, C2_and_dC2, C3_and_dC3] = _converter.convert(_constants);
  const auto & [C1, dC1] = C1_and_dC1;
  const auto & [C2, dC2] = C2_and_dC2;
  const auto & [C3, dC3] = C3_and_dC3;

  const auto I1 = SSR4::identity_C1(C1.options());
  const auto I2 = SSR4::identity_C2(C2.options());
  const auto I3 = SSR4::identity_C3(C3.options());

  if (out)
    _p = C1 * I1 + C2 * I2 + C3 * I3;

  if (dout_din)
  {
    if (const auto * const p1 = nl_param(neml2::name(_constant_types[0])))
      _p.d(*p1) = dC1[0] * I1 + dC2[0] * I2 + dC3[0] * I3;

    if (const auto * const p2 = nl_param(neml2::name(_constant_types[1])))
      _p.d(*p2) = dC1[1] * I1 + dC2[1] * I2 + dC3[1] * I3;

    if (const auto * const p3 = nl_param(neml2::name(_constant_types[2])))
      _p.d(*p3) = dC1[2] * I1 + dC2[2] * I2 + dC3[2] * I3;
  }
}

} // namespace neml2
