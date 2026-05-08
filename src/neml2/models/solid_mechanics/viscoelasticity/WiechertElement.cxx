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

#include "neml2/models/solid_mechanics/viscoelasticity/WiechertElement.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(WiechertElement);

OptionSet
WiechertElement::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Wiechert (generalized Maxwell) viscoelastic model with two Maxwell branches in parallel "
      "with an equilibrium spring. Total stress is \\f$ \\boldsymbol{\\sigma} = (E_\\infty + E_1 + "
      "E_2) \\boldsymbol{\\varepsilon} - E_1 \\boldsymbol{\\varepsilon}_{v,1} - E_2 "
      "\\boldsymbol{\\varepsilon}_{v,2} \\f$, with each viscous strain evolving according to \\f$ "
      "\\dot{\\boldsymbol{\\varepsilon}}_{v,i} = E_i (\\boldsymbol{\\varepsilon} - "
      "\\boldsymbol{\\varepsilon}_{v,i}) / \\eta_i \\f$.";

  options.add_input("strain", "Total strain");
  options.add_input("viscous_strain_1", "Viscous strain in the first Maxwell branch");
  options.add_input("viscous_strain_2", "Viscous strain in the second Maxwell branch");
  options.add_output("stress", "Total stress");
  options.add_parameter<Scalar>("equilibrium_modulus", "Equilibrium spring modulus");
  options.add_parameter<Scalar>("modulus_1", "Spring modulus of the first Maxwell branch");
  options.add_parameter<Scalar>("viscosity_1", "Dashpot viscosity of the first Maxwell branch");
  options.add_parameter<Scalar>("modulus_2", "Spring modulus of the second Maxwell branch");
  options.add_parameter<Scalar>("viscosity_2", "Dashpot viscosity of the second Maxwell branch");

  return options;
}

WiechertElement::WiechertElement(const OptionSet & options)
  : Model(options),
    _E(declare_input_variable<SR2>("strain")),
    _Ev1(declare_input_variable<SR2>("viscous_strain_1")),
    _Ev2(declare_input_variable<SR2>("viscous_strain_2")),
    _S(declare_output_variable<SR2>("stress")),
    _Ev1_dot(declare_output_variable<SR2>(rate_name(_Ev1.name()))),
    _Ev2_dot(declare_output_variable<SR2>(rate_name(_Ev2.name()))),
    _Einf(declare_parameter<Scalar>("Einf", "equilibrium_modulus", true)),
    _E1(declare_parameter<Scalar>("E1", "modulus_1", true)),
    _eta1(declare_parameter<Scalar>("eta1", "viscosity_1", true)),
    _E2(declare_parameter<Scalar>("E2", "modulus_2", true)),
    _eta2(declare_parameter<Scalar>("eta2", "viscosity_2", true))
{
}

void
WiechertElement::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto diff1 = _E - _Ev1;
  auto diff2 = _E - _Ev2;
  auto S1 = _E1 * diff1;
  auto S2 = _E2 * diff2;

  if (out)
  {
    _S = _Einf * _E + S1 + S2;
    _Ev1_dot = S1 / _eta1;
    _Ev2_dot = S2 / _eta2;
  }

  if (dout_din)
  {
    auto I = imap_v<SR2>(_E.options());
    _S.d(_E) = (_Einf + _E1 + _E2) * I;
    _S.d(_Ev1) = -_E1 * I;
    _S.d(_Ev2) = -_E2 * I;

    _Ev1_dot.d(_E) = (_E1 / _eta1) * I;
    _Ev1_dot.d(_Ev1) = -(_E1 / _eta1) * I;

    _Ev2_dot.d(_E) = (_E2 / _eta2) * I;
    _Ev2_dot.d(_Ev2) = -(_E2 / _eta2) * I;

    if (const auto * const Einf = nl_param("Einf"))
      _S.d(*Einf) = _E();

    if (const auto * const E1 = nl_param("E1"))
    {
      _S.d(*E1) = diff1;
      _Ev1_dot.d(*E1) = diff1 / _eta1;
    }

    if (const auto * const eta1 = nl_param("eta1"))
      _Ev1_dot.d(*eta1) = -S1 / (_eta1 * _eta1);

    if (const auto * const E2 = nl_param("E2"))
    {
      _S.d(*E2) = diff2;
      _Ev2_dot.d(*E2) = diff2 / _eta2;
    }

    if (const auto * const eta2 = nl_param("eta2"))
      _Ev2_dot.d(*eta2) = -S2 / (_eta2 * _eta2);
  }
}
} // namespace neml2
