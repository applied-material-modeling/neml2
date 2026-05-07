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

#include "neml2/models/solid_mechanics/viscoelasticity/ZenerViscoelasticity.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(ZenerViscoelasticity);

OptionSet
ZenerViscoelasticity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Zener (Standard Linear Solid) viscoelastic model: an equilibrium spring in parallel with a "
      "Maxwell branch. The total stress is \\f$ \\boldsymbol{\\sigma} = (E_\\infty + E_M) "
      "\\boldsymbol{\\varepsilon} - E_M \\boldsymbol{\\varepsilon}_v \\f$, and the Maxwell-branch "
      "viscous strain evolves as \\f$ \\dot{\\boldsymbol{\\varepsilon}}_v = E_M "
      "(\\boldsymbol{\\varepsilon} - \\boldsymbol{\\varepsilon}_v) / \\eta_M \\f$.";

  options.add_input("strain", "Total strain");
  options.add_input("viscous_strain", "Viscous strain in the Maxwell branch");
  options.add_output("stress", "Total stress");
  options.add_parameter<Scalar>("equilibrium_modulus", "Equilibrium spring modulus");
  options.add_parameter<Scalar>("maxwell_modulus", "Maxwell branch spring modulus");
  options.add_parameter<Scalar>("maxwell_viscosity", "Maxwell branch dashpot viscosity");

  return options;
}

ZenerViscoelasticity::ZenerViscoelasticity(const OptionSet & options)
  : Model(options),
    _E(declare_input_variable<SR2>("strain")),
    _Ev(declare_input_variable<SR2>("viscous_strain")),
    _S(declare_output_variable<SR2>("stress")),
    _Ev_dot(declare_output_variable<SR2>(rate_name(_Ev.name()))),
    _Einf(declare_parameter<Scalar>("Einf", "equilibrium_modulus", true)),
    _EM(declare_parameter<Scalar>("EM", "maxwell_modulus", true)),
    _etaM(declare_parameter<Scalar>("etaM", "maxwell_viscosity", true))
{
}

void
ZenerViscoelasticity::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Maxwell branch stress: sigma_M = E_M * (E - Ev)
  auto Ediff = _E - _Ev;
  auto SM = _EM * Ediff;

  if (out)
  {
    _S = _Einf * _E + SM;
    _Ev_dot = SM / _etaM;
  }

  if (dout_din)
  {
    auto I = imap_v<SR2>(_E.options());
    _S.d(_E) = (_Einf + _EM) * I;
    _S.d(_Ev) = -_EM * I;
    _Ev_dot.d(_E) = (_EM / _etaM) * I;
    _Ev_dot.d(_Ev) = -(_EM / _etaM) * I;

    if (const auto * const Einf = nl_param("Einf"))
      _S.d(*Einf) = _E();

    if (const auto * const EM = nl_param("EM"))
    {
      _S.d(*EM) = Ediff;
      _Ev_dot.d(*EM) = Ediff / _etaM;
    }

    if (const auto * const etaM = nl_param("etaM"))
      _Ev_dot.d(*etaM) = -SM / (_etaM * _etaM);
  }
}
} // namespace neml2
