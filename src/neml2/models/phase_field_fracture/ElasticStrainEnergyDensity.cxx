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

#include "neml2/models/phase_field_fracture/ElasticStrainEnergyDensity.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(ElasticStrainEnergyDensity);

OptionSet
ElasticStrainEnergyDensity::expected_options()
{
  OptionSet options = LinearIsotropicElasticity::expected_options();
  options.doc() =
      "Calculates elastic strain energy density based on linear elastic isotropic response";
  options.set_output("elastic_strain_energy") = VariableName(STATE, "psie");
  return options;
}

ElasticStrainEnergyDensity::ElasticStrainEnergyDensity(const OptionSet & options)
  : LinearIsotropicElasticity(options),
  _psie(declare_output_variable<Scalar>("elastic_strain_energy"))

{
}

void
ElasticStrainEnergyDensity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const auto [K_and_dK, G_and_dG] = _converter.convert(_constants);
  const auto & [K, dK] = K_and_dK;
  const auto & [G, dG] = G_and_dG;
  const auto vf = _compliance ? 1 / (3 * K) : 3 * K;
  const auto df = _compliance ? 1 / (2 * G) : 2 * G;

  _to = vf * SR2(_from).vol() + df * SR2(_from).dev();

  if (out)
    
    _psie = 0.5 * SR2(_to).inner(_from);
    
  if (dout_din)
  {
    
    _psie.d(_from) = _to;
    
  }
  if (d2out_din2)
  {

    const auto I = SSR4::identity_vol(_from.options());
    const auto J = SSR4::identity_dev(_from.options());

    _psie.d(_from, _from) = vf * I + df * J;

  }
}
} // namespace neml2
