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

#include "neml2/models/kwn/KineticFactor.h"

#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(KineticFactor);

OptionSet
KineticFactor::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the nucleation kinetic factor.";

  options.add_input("critical_radius", "Critical radius for nucleation");

  options.add_input("projected_diffusivity_sum", "Projected diffusivity sum");

  options.add_parameter<Scalar>("molar_volume", "Molar volume of the precipitate");

  options.add_parameter<Scalar>("avogadro_number", "Avogadro's number");

  options.add_output("kinetic_factor", "Kinetic factor for nucleation");

  return options;
}

KineticFactor::KineticFactor(const OptionSet & options)
  : Model(options),
    _R_crit(declare_input_variable<Scalar>("critical_radius")),
    _sum(declare_input_variable<Scalar>("projected_diffusivity_sum")),
    _V_m(declare_parameter<Scalar>("V_m", "molar_volume")),
    _N_a(declare_parameter<Scalar>("N_a", "avogadro_number")),
    _beta(declare_output_variable<Scalar>("kinetic_factor"))
{
}

void
KineticFactor::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto R_crit = _R_crit();
  const auto sum = _sum();
  const auto V_m = _V_m;
  const auto N_a = _N_a;

  const auto coef = 4.0 * neml2::pi * pow(N_a, 4.0 / 3.0) / pow(V_m, 4.0 / 3.0);
  const auto beta = coef * R_crit * R_crit / sum;

  if (out)
    _beta = beta;

  if (dout_din)
  {
    _beta.d(_R_crit) = 2.0 * beta / R_crit;
    _beta.d(_sum) = -beta / sum;
  }
}
} // namespace neml2
