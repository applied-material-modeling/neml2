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

#include "neml2/models/solid_mechanics/traction_separation_law/AlfanoCrisfieldTraction.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(AlfanoCrisfieldTraction);

OptionSet
AlfanoCrisfieldTraction::expected_options()
{
  OptionSet options = BilinearMixedModeTraction::expected_options();
  options.doc() =
      "Bilinear mixed-mode cohesive-zone traction-separation law of Alfano & Crisfield (2001) "
      "with the power-law mixed-mode propagation criterion. Damage is irreversible. The normal "
      "compressive branch is restored through a Macaulay split so interpenetration does not "
      "soften the interface. Variable derivatives are hand-coded.";
  return options;
}

AlfanoCrisfieldTraction::AlfanoCrisfieldTraction(const OptionSet & options)
  : BilinearMixedModeTraction(options)
{
}

AlfanoCrisfieldTraction::DeltaFinalResult
AlfanoCrisfieldTraction::compute_delta_final(const DeltaFinalContext & ctx, bool dout_din) const
{
  // Power-law criterion in the opening branch:
  //   delta_final = (2 + 2*beta^2)/(K*delta_init) * Gc_mixed^(-1/eta)
  //   Gc_mixed    = (1/GIc)^eta + (beta^2/GIIc)^eta
  const auto pow_base_PL = ctx.beta_sq / _GIIc + ctx.eps;
  const auto Gc_mixed = neml2::pow(1.0 / _GIc, _eta) + neml2::pow(pow_base_PL, _eta);
  const auto Gc_term = neml2::pow(Gc_mixed, -1.0 / _eta);
  const auto prefactor = (2.0 + 2.0 * ctx.beta_sq) / ctx.Kdelta_init_mixed;
  const auto delta_final_mixed = prefactor * Gc_term;

  if (!dout_din)
    return {delta_final_mixed, Vec::zeros_like(ctx.dbeta_ddelta_open)};

  const auto dprefactor_dbeta = 4.0 * ctx.beta / ctx.Kdelta_init_mixed;
  const auto ddelta_final_ddelta_init_PL = -prefactor / ctx.delta_init_mixed * Gc_term;
  // dGc_mixed/dbeta = eta * (beta^2/GIIc + eps)^(eta-1) * (2*beta/GIIc)
  const auto dGc_mixed_dbeta =
      _eta * neml2::pow(pow_base_PL, _eta - 1.0) * (2.0 * ctx.beta / _GIIc);
  // dGc_term/dbeta = (-1/eta) * Gc_mixed^(-1/eta - 1) * dGc_mixed/dbeta
  const auto dGc_term_dbeta =
      (-1.0 / _eta) * neml2::pow(Gc_mixed, -1.0 / _eta - 1.0) * dGc_mixed_dbeta;
  const auto ddelta_final_dbeta_PL = dprefactor_dbeta * Gc_term + prefactor * dGc_term_dbeta;

  const auto ddelta_open = ddelta_final_ddelta_init_PL * ctx.ddelta_init_ddelta_open +
                           ddelta_final_dbeta_PL * ctx.dbeta_ddelta_open;
  return {delta_final_mixed, ddelta_open};
}
} // namespace neml2
