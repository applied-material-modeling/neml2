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
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/macaulay.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/linalg/eigh.h"
#include "neml2/tensors/functions/linalg/ieigh.h"
#include "neml2/tensors/functions/linalg/dsptrf.h"
#include "neml2/base/EnumSelection.h"

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

  EnumSelection type_selection({"NONE", "SPECTRAL", "VOLDEV"}, "NONE");
  options.set<EnumSelection>("decomposition") = type_selection;
  options.set("decomposition").doc() =
      "Strain energy density decomposition types, options are: " + type_selection.candidates_str();

  return options;
}

LinearIsotropicStrainEnergyDensity::LinearIsotropicStrainEnergyDensity(const OptionSet & options)
  : ElasticityInterface<StrainEnergyDensity, 2>(options),
    _converter(_constant_types, _need_derivs),
    _decomposition(options.get<EnumSelection>("decomposition").as<DecompositionType>())
{
}

void
LinearIsotropicStrainEnergyDensity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  switch (_decomposition)
  {
    case DecompositionType::NONE:
      no_decomposition(out, dout_din, d2out_din2);
      break;
    case DecompositionType::VOLDEV:
      voldev_decomposition(out, dout_din, d2out_din2);
      break;
    case DecompositionType::SPECTRAL:
      spectral_decomposition(out, dout_din, d2out_din2);
      break;
    default:
      throw NEMLException("LinearIsotropicStrainEnergyDensity: Unsupported decomposition type.");
  }
}

void
LinearIsotropicStrainEnergyDensity::no_decomposition(bool out, bool dout_din, bool d2out_din2)
{
  const auto [K_and_dK, G_and_dG] = _converter.convert(_constants);
  const auto & [K, dK] = K_and_dK;
  const auto & [G, dG] = G_and_dG;
  const auto etr = SR2(_strain).tr();
  const auto edev = SR2(_strain).dev();
  const auto I2 = SR2::identity(_strain.options());
  const auto I4 = SSR4::identity(_strain.options());
  const auto J = SSR4::identity_dev(_strain.options());

  if (out)
  {
    _psie_active = 0.5 * K * etr * etr + G * edev.inner(edev);
    _psie_inactive = Scalar::zeros_like(_psie_active);
  }
  if (dout_din)
  {
    _psie_active.d(_strain) = K * etr * I2 + 2 * G * edev;
  }
  if (d2out_din2)
  {
    _psie_active.d(_strain, _strain) = K * I4 + 2 * G * J;
  }
}

void
LinearIsotropicStrainEnergyDensity::voldev_decomposition(bool out, bool dout_din, bool d2out_din2)
{
  const auto [K_and_dK, G_and_dG] = _converter.convert(_constants);
  const auto & [K, dK] = K_and_dK;
  const auto & [G, dG] = G_and_dG;
  const auto etr = SR2(_strain).tr();
  const auto edev = SR2(_strain).dev();
  const auto I2 = SR2::identity(_strain.options());
  const auto I4 = SSR4::identity(_strain.options());
  const auto J = SSR4::identity_dev(_strain.options());

  // Decompose based on the trace of strain
  const auto etr_pos = macaulay(etr);
  const auto etr_neg = etr - etr_pos;

  if (out)
  {
    _psie_active = 0.5 * K * etr_pos * etr_pos + G * edev.inner(edev);
    _psie_inactive = 0.5 * K * etr_neg * etr_neg;
  }
  if (dout_din)
  {
    _psie_active.d(_strain) = K * etr_pos * I2 + 2 * G * edev;
    _psie_inactive.d(_strain) = K * etr_neg * I2;
  }
  if (d2out_din2)
  {
    _psie_active.d(_strain, _strain) = K * heaviside(etr) * I4 + 2 * G * J;
    _psie_inactive.d(_strain, _strain) = K * heaviside(-etr) * I4;
  }
}

void
LinearIsotropicStrainEnergyDensity::spectral_decomposition(bool out, bool dout_din, bool d2out_din2)
{
  const auto [lambda_and_dlambda, G_and_dG] = _converter.convert(_constants);
  const auto & [lambda, dlambda] = lambda_and_dlambda;
  const auto & [G, dG] = G_and_dG;
  const auto etr = SR2(_strain).tr();
  const auto edev = SR2(_strain).dev();
  const auto I2 = SR2::identity(_strain.options());
  const auto I4 = SSR4::identity(_strain.options());

  // Decompose based on the eigenvalues of strain
  const auto etr_pos = macaulay(etr);
  const auto etr_neg = etr - etr_pos;
  const auto [evals, evecs] = linalg::eigh(_strain);
  const auto evals_pos = macaulay(evals);
  const auto e_pos = linalg::ieigh(evals_pos, evecs);
  const auto e_neg = _strain - e_pos;

  if (out)
  {
    _psie_active = 0.5 * lambda * etr_pos * etr_pos + G * e_pos.inner(e_pos);
    _psie_inactive = 0.5 * lambda * etr_neg * etr_neg + G * e_neg.inner(e_neg);
  }
  if (dout_din)
  {
    _psie_active.d(_strain) = lambda * etr_pos * I2 + 2 * G * e_pos;
    _psie_inactive.d(_strain) = lambda * etr_neg * I2 + 2 * G * e_neg;
  }
  if (d2out_din2)
  {
    const auto P4_pos = linalg::dsptrf(evals, evecs, evals_pos, heaviside(evals));
    const auto P4_neg = SSR4::identity_sym(_strain.options()) - P4_pos;
    _psie_active.d(_strain, _strain) = lambda * heaviside(etr) * I4 + 2 * G * P4_pos;
    _psie_inactive.d(_strain, _strain) = lambda * heaviside(-etr) * I4 + 2 * G * P4_neg;
  }
}

} // namespace neml2
