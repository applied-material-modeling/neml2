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

#include "neml2/models/solid_mechanics/viscoelasticity/BurgersViscoelasticity.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(BurgersViscoelasticity);

OptionSet
BurgersViscoelasticity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Burgers viscoelastic model: a Maxwell element in series with a Kelvin-Voigt element. The "
      "shared stress is \\f$ \\boldsymbol{\\sigma} = E_M (\\boldsymbol{\\varepsilon} - "
      "\\boldsymbol{\\varepsilon}_{v,M} - \\boldsymbol{\\varepsilon}_K) \\f$, and the internal "
      "strains evolve as \\f$ \\dot{\\boldsymbol{\\varepsilon}}_{v,M} = \\boldsymbol{\\sigma}/"
      "\\eta_M \\f$ and \\f$ \\dot{\\boldsymbol{\\varepsilon}}_K = (\\boldsymbol{\\sigma} - E_K "
      "\\boldsymbol{\\varepsilon}_K) / \\eta_K \\f$.";

  options.add_input("strain", "Total strain");
  options.add_input("maxwell_viscous_strain", "Viscous strain in the Maxwell branch dashpot");
  options.add_input("kelvin_voigt_strain", "Strain in the Kelvin-Voigt branch");
  options.add_output("stress", "Total stress (shared between Maxwell and Kelvin-Voigt elements)");
  options.add_output("maxwell_viscous_strain_rate",
                     "Rate of viscous strain in the Maxwell branch dashpot");
  options.add_output("kelvin_voigt_strain_rate", "Rate of strain in the Kelvin-Voigt branch");
  options.add_parameter<Scalar>("maxwell_modulus", "Maxwell branch spring modulus");
  options.add_parameter<Scalar>("maxwell_viscosity", "Maxwell branch dashpot viscosity");
  options.add_parameter<Scalar>("kelvin_modulus", "Kelvin-Voigt branch spring modulus");
  options.add_parameter<Scalar>("kelvin_viscosity", "Kelvin-Voigt branch dashpot viscosity");

  return options;
}

BurgersViscoelasticity::BurgersViscoelasticity(const OptionSet & options)
  : Model(options),
    _E(declare_input_variable<SR2>("strain")),
    _EvM(declare_input_variable<SR2>("maxwell_viscous_strain")),
    _EK(declare_input_variable<SR2>("kelvin_voigt_strain")),
    _S(declare_output_variable<SR2>("stress")),
    _EvM_dot(declare_output_variable<SR2>("maxwell_viscous_strain_rate")),
    _EK_dot(declare_output_variable<SR2>("kelvin_voigt_strain_rate")),
    _EM(declare_parameter<Scalar>("EM", "maxwell_modulus", true)),
    _etaM(declare_parameter<Scalar>("etaM", "maxwell_viscosity", true)),
    _EK_param(declare_parameter<Scalar>("EK", "kelvin_modulus", true)),
    _etaK(declare_parameter<Scalar>("etaK", "kelvin_viscosity", true))
{
}

void
BurgersViscoelasticity::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Shared series stress: sigma = E_M * (E - EvM - EK)
  auto Eel = _E - _EvM - _EK;
  auto S = _EM * Eel;

  if (out)
  {
    _S = S;
    _EvM_dot = S / _etaM;
    _EK_dot = (S - _EK_param * _EK) / _etaK;
  }

  if (dout_din)
  {
    auto I = imap_v<SR2>(_E.options());

    _S.d(_E) = _EM * I;
    _S.d(_EvM) = -_EM * I;
    _S.d(_EK) = -_EM * I;

    _EvM_dot.d(_E) = (_EM / _etaM) * I;
    _EvM_dot.d(_EvM) = -(_EM / _etaM) * I;
    _EvM_dot.d(_EK) = -(_EM / _etaM) * I;

    _EK_dot.d(_E) = (_EM / _etaK) * I;
    _EK_dot.d(_EvM) = -(_EM / _etaK) * I;
    _EK_dot.d(_EK) = -((_EM + _EK_param) / _etaK) * I;

    if (const auto * const EM = nl_param("EM"))
    {
      _S.d(*EM) = Eel;
      _EvM_dot.d(*EM) = Eel / _etaM;
      _EK_dot.d(*EM) = Eel / _etaK;
    }

    if (const auto * const etaM = nl_param("etaM"))
      _EvM_dot.d(*etaM) = -S / (_etaM * _etaM);

    if (const auto * const EK = nl_param("EK"))
      _EK_dot.d(*EK) = -_EK() / _etaK;

    if (const auto * const etaK = nl_param("etaK"))
      _EK_dot.d(*etaK) = -(S - _EK_param * _EK) / (_etaK * _etaK);
  }
}
} // namespace neml2
