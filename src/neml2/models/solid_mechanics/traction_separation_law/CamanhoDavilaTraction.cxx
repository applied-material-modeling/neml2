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

#include "neml2/models/solid_mechanics/traction_separation_law/CamanhoDavilaTraction.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(CamanhoDavilaTraction);

OptionSet
CamanhoDavilaTraction::expected_options()
{
  OptionSet options = BilinearMixedModeTraction::expected_options();
  options.doc() =
      "Bilinear mixed-mode cohesive-zone traction-separation law of Camanho & Davila (2002) "
      "with the Benzeggagh-Kenane (BK) propagation criterion. Damage is irreversible. The "
      "normal compressive branch is restored through a Macaulay split so interpenetration does "
      "not soften the interface. Variable derivatives are hand-coded.";
  return options;
}

CamanhoDavilaTraction::CamanhoDavilaTraction(const OptionSet & options)
  : BilinearMixedModeTraction(options)
{
}

CamanhoDavilaTraction::DeltaFinalResult
CamanhoDavilaTraction::compute_delta_final(const DeltaFinalContext & ctx, bool dout_din) const
{
  // BK criterion in the opening branch:
  //   delta_final = 2/(K*delta_init) * [GIc + (GIIc - GIc) * (beta^2/(1+beta^2))^eta]
  const auto beta_sq_ratio = ctx.beta_sq / (1.0 + ctx.beta_sq);
  const auto pow_base_BK = beta_sq_ratio + ctx.eps;
  const auto term = _GIc + (_GIIc - _GIc) * neml2::pow(pow_base_BK, _eta);
  const auto delta_final_mixed = 2.0 / ctx.Kdelta_init_mixed * term;

  if (!dout_din)
    return {delta_final_mixed, Vec::zeros_like(ctx.dbeta_ddelta_open)};

  // d(delta_final)/d(delta_init) = -delta_final / delta_init
  const auto ddelta_final_ddelta_init_BK = -delta_final_mixed / ctx.delta_init_mixed;
  // d(beta_sq_ratio)/d(beta) = 2*beta / (1+beta^2)^2
  const auto one_plus_beta_sq = 1.0 + ctx.beta_sq;
  const auto dbeta_sq_ratio_dbeta = 2.0 * ctx.beta / (one_plus_beta_sq * one_plus_beta_sq);
  // d(delta_final)/d(beta) = (2/(K*delta_init)) * (GIIc - GIc) * eta * (r+eps)^(eta-1) * dr/dbeta
  const auto ddelta_final_dbeta_BK = (2.0 / ctx.Kdelta_init_mixed) * (_GIIc - _GIc) * _eta *
                                     neml2::pow(pow_base_BK, _eta - 1.0) * dbeta_sq_ratio_dbeta;

  const auto ddelta_open = ddelta_final_ddelta_init_BK * ctx.ddelta_init_ddelta_open +
                           ddelta_final_dbeta_BK * ctx.dbeta_ddelta_open;
  return {delta_final_mixed, ddelta_open};
}
} // namespace neml2
