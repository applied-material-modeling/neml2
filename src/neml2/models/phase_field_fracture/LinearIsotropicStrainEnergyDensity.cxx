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

#include "neml2/models/phase_field_fracture/LinearIsotropicStrainEnergyDensity.h"
#include "neml2/misc/errors.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/macaulay.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(LinearIsotropicStrainEnergyDensity);

OptionSet
LinearIsotropicStrainEnergyDensity::expected_options()
{
  OptionSet options = ElasticityInterface<StrainEnergyDensity, 2>::expected_options();
  options.doc() =
      "Calculates elastic strain energy density based on linear elastic isotropic response";
  options.set<bool>("define_second_derivatives") = true;

  return options;
}

LinearIsotropicStrainEnergyDensity::LinearIsotropicStrainEnergyDensity(const OptionSet & options)
  : ElasticityInterface<StrainEnergyDensity, 2>(options),
    _converter(_constant_types, _need_derivs)
{
}

void
LinearIsotropicStrainEnergyDensity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const auto [K_and_dK, G_and_dG] = _converter.convert(_constants);
  const auto & [K, dK] = K_and_dK;
  const auto & [G, dG] = G_and_dG;
  const auto vf = 3 * K;
  const auto df = 2 * G;

  const auto s = vf * SR2(_strain).vol() + df * SR2(_strain).dev();

  if (_decomposition == DecompositionType::NONE)
  {
    if (out)
    {
      _psie_active = 0.5 * SR2(s).inner(_strain);
      _psie_inactive = Scalar::create(0.0, _strain.options());
    }
    if (dout_din)
    {
      _psie_active.d(_strain) = s;
      _psie_inactive.d(_strain) = SR2::fill(0.0, s.options());
    }
    if (d2out_din2)
    {
      const auto I = SSR4::identity_vol(_strain.options());
      const auto J = SSR4::identity_dev(_strain.options());

      _psie_active.d(_strain, _strain) = vf * I + df * J;
      _psie_inactive.d(_strain, _strain) = 0.0 * I + 0.0 * J;
    }
  }
  else if (_decomposition == DecompositionType::VOLDEV)
  {
    const auto I2 = SR2::identity(_strain.options());
    auto strain_trace = SR2(_strain).tr();
    auto strain_trace_pos = macaulay(strain_trace);
    auto strain_trace_neg = strain_trace - strain_trace_pos;
    auto strain_dev = SR2(_strain).dev();

    if (out)
    {
      auto psie_intact =
          0.5 * K * strain_trace * strain_trace + G * SR2(strain_dev).inner(strain_dev);
      _psie_inactive = 0.5 * K * strain_trace_neg * strain_trace_neg;
      _psie_active = psie_intact - _psie_inactive;
    }
    if (dout_din)
    {
      auto s_intact = K * strain_trace * I2 + 2 * G * strain_dev;
      _psie_inactive.d(_strain) = K * strain_trace_neg * I2;
      _psie_active.d(_strain) = s_intact - K * strain_trace_neg * I2;
    }
    if (d2out_din2)
    {
      const auto I = SSR4::identity_vol(_strain.options());
      const auto J = SSR4::identity_dev(_strain.options());
      const auto elasticity_tensor = vf * I + df * J;

      auto multiplier = where(strain_trace_neg < 0,
                              Scalar::ones_like(strain_trace_neg),
                              Scalar::zeros_like(strain_trace_neg));

      auto dstressneg_dstrain = multiplier * (vf * I);
      _psie_inactive.d(_strain, _strain) = dstressneg_dstrain;
      _psie_active.d(_strain, _strain) = elasticity_tensor - dstressneg_dstrain;
    }
  }
  else
    throw NEMLException("LinearIsotropicStrainEnergyDensity: Unsupported decomposition type.");
}
} // namespace neml2
