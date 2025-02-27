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

#include "neml2/models/solid_mechanics/elasticity/IsotropicElasticityTensor.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(IsotropicElasticityTensor);

OptionSet
IsotropicElasticityTensor::expected_options()
{
  OptionSet options = ElasticityInterface<Model, 2>::expected_options();
  options.doc() = "This class defines an isotropic elasticity tensor using two parameters."
                  "  Various options are available for which two parameters to provide.";

  return options;
}

IsotropicElasticityTensor::IsotropicElasticityTensor(const OptionSet & options)
  : ElasticityInterface<Model, 2>(options),
    _converter(_constant_types, _need_derivs),
    _C(declare_output_variable<SSR4>(VariableName(PARAMETERS, name())))
{
}

void
IsotropicElasticityTensor::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto [K_and_dK, G_and_dG] = _converter.convert(_constants);
  const auto & [K, dK] = K_and_dK;
  const auto & [G, dG] = G_and_dG;

  const auto Iv = SSR4::identity_vol(K.options());
  const auto Id = SSR4::identity_dev(G.options());

  if (out)
    _C = 3.0 * K * Iv + 2.0 * G * Id;

  if (dout_din)
  {
    if (const auto * const p1 = nl_param(neml2::name(_constant_types[0])))
      _C.d(*p1) = 3.0 * dK[0] * Iv + 2.0 * dG[0] * Id;

    if (const auto * const p2 = nl_param(neml2::name(_constant_types[1])))
      _C.d(*p2) = 3.0 * dK[1] * Iv + 2.0 * dG[1] * Id;
  }
}

} // namespace neml2
