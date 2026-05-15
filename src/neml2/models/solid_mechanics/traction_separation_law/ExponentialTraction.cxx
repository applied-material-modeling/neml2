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

#include "neml2/models/solid_mechanics/traction_separation_law/ExponentialTraction.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(ExponentialTraction);

OptionSet
ExponentialTraction::expected_options()
{
  OptionSet options = TractionSeparationLaw::expected_options();
  options.doc() =
      "Exponential cohesive-zone traction-separation law with isotropic scalar damage. "
      "An effective scalar separation \\f$ \\delta_{\\text{eff}} = "
      "\\sqrt{\\delta_n^2 + \\beta(\\delta_{s1}^2 + \\delta_{s2}^2) + \\varepsilon} \\f$ drives "
      "an exponential damage \\f$ d = 1 - \\exp(-\\kappa/\\delta_0) \\f$, and the traction is "
      "\\f$ \\boldsymbol{T} = (1 - d)\\,(G_c/\\delta_0^2)\\,\\boldsymbol{\\delta} \\f$. "
      "When `irreversible_damage` is true, \\f$ \\kappa \\f$ is the historical maximum of "
      "\\f$ \\delta_{\\text{eff}} \\f$ (no healing); otherwise it tracks the current value.";

  options.add_parameter<Scalar>("fracture_toughness", "Fracture toughness G_c");
  options.add_parameter<Scalar>("characteristic_length", "Softening length scale delta_0");
  options.add_parameter<Scalar>("tangential_weight",
                                "Tangential weighting factor beta in the effective separation");

  options.add<bool>(
      "irreversible_damage",
      true,
      "If true, freeze damage at the historical peak effective separation (no healing). "
      "If false, damage tracks the current effective separation reversibly.");

  return options;
}

ExponentialTraction::ExponentialTraction(const OptionSet & options)
  : TractionSeparationLaw(options),
    _kappa(declare_output_variable<Scalar>("effective_displacement_jump_max")),
    _kappa_n(declare_input_variable<Scalar>(history_name(_kappa.name(), /*nstep=*/1))),
    // allow_nonlinear=false: this model does not provide _T.d(*Gc) / etc., so the parameters
    // must be declared as plain (non-nonlinear) parameters.
    _Gc(declare_parameter<Scalar>("Gc", "fracture_toughness", false)),
    _delta0(declare_parameter<Scalar>("delta0", "characteristic_length", false)),
    _beta(declare_parameter<Scalar>("beta", "tangential_weight", false)),
    _irreversible(options.get<bool>("irreversible_damage"))
{
}

void
ExponentialTraction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Component access on the underlying Vec
  const auto delta_n = _delta()(0);
  const auto delta_s1 = _delta()(1);
  const auto delta_s2 = _delta()(2);

  // Effective scalar separation, regularized with the dtype's machine epsilon to avoid the AD
  // singularity of sqrt() at zero jump.
  const auto eps = machine_precision(_delta.scalar_type());
  const auto delta_eff_sq = delta_n * delta_n + _beta * (delta_s1 * delta_s1 + delta_s2 * delta_s2);
  const auto delta_eff = neml2::sqrt(delta_eff_sq + eps);

  // Damage driver: irreversible -> max(kappa_n, delta_eff); reversible -> delta_eff
  const auto kappa = _irreversible ? neml2::where(delta_eff > _kappa_n(), delta_eff, _kappa_n())
                                   : Scalar(delta_eff);

  // Damage and degraded stiffness scalar
  const auto d = Scalar::ones_like(_Gc) - neml2::exp(-kappa / _delta0);
  const auto c = _Gc / (_delta0 * _delta0);
  const auto factor = (Scalar::ones_like(_Gc) - d) * c;

  if (out)
  {
    _T = Vec::fill(factor * delta_n, factor * delta_s1, factor * delta_s2);
    _kappa = kappa;
  }

  if (dout_din)
  {
    // d delta_eff / d delta_j
    const auto dde_ddn = delta_n / delta_eff;
    const auto dde_dds1 = _beta * delta_s1 / delta_eff;
    const auto dde_dds2 = _beta * delta_s2 / delta_eff;

    // d kappa / d input. In irreversible mode, kappa freezes when delta_eff <= kappa_n.
    const auto zero_s = Scalar::zeros_like(_Gc);
    const auto one_s = Scalar::ones_like(_Gc);

    Scalar dk_ddn = dde_ddn;
    Scalar dk_dds1 = dde_dds1;
    Scalar dk_dds2 = dde_dds2;
    Scalar dk_dkn = zero_s;
    if (_irreversible)
    {
      const auto advancing = delta_eff > _kappa_n();
      dk_ddn = neml2::where(advancing, dde_ddn, zero_s);
      dk_dds1 = neml2::where(advancing, dde_dds1, zero_s);
      dk_dds2 = neml2::where(advancing, dde_dds2, zero_s);
      dk_dkn = neml2::where(advancing, zero_s, one_s);
    }

    // d d / d kappa = (1/delta_0) * exp(-kappa/delta_0)
    const auto dd_dkappa = neml2::exp(-kappa / _delta0) / _delta0;

    // d T_i / d delta_j = (1-d) c delta_ij - c delta_i (dd/dkappa)(dkappa/d delta_j)
    const auto cdd_ddn = c * dd_dkappa * dk_ddn;
    const auto cdd_dds1 = c * dd_dkappa * dk_dds1;
    const auto cdd_dds2 = c * dd_dkappa * dk_dds2;

    const auto J11 = factor - cdd_ddn * delta_n;
    const auto J12 = -cdd_dds1 * delta_n;
    const auto J13 = -cdd_dds2 * delta_n;
    const auto J21 = -cdd_ddn * delta_s1;
    const auto J22 = factor - cdd_dds1 * delta_s1;
    const auto J23 = -cdd_dds2 * delta_s1;
    const auto J31 = -cdd_ddn * delta_s2;
    const auto J32 = -cdd_dds1 * delta_s2;
    const auto J33 = factor - cdd_dds2 * delta_s2;

    _T.d(_delta) = R2::fill(J11, J12, J13, J21, J22, J23, J31, J32, J33);

    // d T_i / d kappa_n = -c delta_i (dd/dkappa)(dkappa/d kappa_n)
    const auto cdd_dkn = c * dd_dkappa * dk_dkn;
    _T.d(_kappa_n) = Vec::fill(-cdd_dkn * delta_n, -cdd_dkn * delta_s1, -cdd_dkn * delta_s2);

    // d kappa / d delta_j and d kappa / d kappa_n
    _kappa.d(_delta) = Vec::fill(dk_ddn, dk_dds1, dk_dds2);
    _kappa.d(_kappa_n) = dk_dkn;
  }
}
} // namespace neml2
