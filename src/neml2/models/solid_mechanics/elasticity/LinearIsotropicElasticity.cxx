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

#include "neml2/models/solid_mechanics/elasticity/LinearIsotropicElasticity.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/dev.h"
#include "neml2/tensors/functions/vol.h"

namespace neml2
{
register_NEML2_object(LinearIsotropicElasticity);

OptionSet
LinearIsotropicElasticity::expected_options()
{
  OptionSet options = ElasticityInterface<Elasticity, 2>::expected_options();
  options.doc() += " for linear isotropic material. \\f$ \\boldsymbol{\\sigma} = K \\tr "
                   "\\boldsymbol{\\varepsilon}_e + 2 G \\text{dev} \\boldsymbol{\\varepsilon}_e "
                   "\\f$, where \\f$ K \\f$ and \\f$ G \\f$ are bulk and shear moduli, "
                   "respectively. Other pairs of Lame parameters are also supported. ";

  return options;
}

LinearIsotropicElasticity::LinearIsotropicElasticity(const OptionSet & options)
  : ElasticityInterface<Elasticity, 2>(options),
    _converter(_constant_types, _need_derivs)
{
}

void
LinearIsotropicElasticity::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto [K_and_dK, G_and_dG] = _converter.convert(_constants);
  const auto & [K, dK] = K_and_dK;
  const auto & [G, dG] = G_and_dG;
  const auto vf = _compliance ? 1 / (3 * K) : 3 * K;
  const auto df = _compliance ? 1 / (2 * G) : 2 * G;

  if (out)
    _to = vf * neml2::vol(_from()) + df * neml2::dev(_from());

  if (dout_din)
  {
    const auto I = SSR4::identity_vol(_from.options());
    const auto J = SSR4::identity_dev(_from.options());

    if (_from.is_dependent())
      _to.d(_from) = vf * I + df * J;

    const auto * const p1 = nl_param(neml2::name(_constant_types[0]));
    const auto * const p2 = nl_param(neml2::name(_constant_types[1]));
    const auto dvf_dK = _compliance ? -vf / K : Scalar::full(3.0, K.options());
    const auto ddf_dG = _compliance ? -df / G : Scalar::full(2.0, G.options());

    if (p1)
      _to.d(*p1) = dvf_dK * dK[0] * neml2::vol(_from()) + ddf_dG * dG[0] * neml2::dev(_from());

    if (p2)
      _to.d(*p2) = dvf_dK * dK[1] * neml2::vol(_from()) + ddf_dG * dG[1] * neml2::dev(_from());
  }
}
} // namespace neml2
