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

#include "neml2/models/solid_mechanics/cohesive/ExpTractionSeparation.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(ExpTractionSeparation);

OptionSet
ExpTractionSeparation::expected_options()
{
  OptionSet options = TractionSeparationModel::expected_options();
  options.doc() +=
      " following an exponential damage law. The interface traction is "
      "\\f$ T_i = \\frac{G_c}{\\delta_0^2} \\exp\\!\\left(-\\frac{\\delta_{\\mathrm{eff}}}"
      "{\\delta_0}\\right) \\delta_i \\f$, "
      "where the effective scalar displacement jump is "
      "\\f$ \\delta_{\\mathrm{eff}} = \\sqrt{\\delta_n^2 + \\beta(\\delta_{s1}^2 + "
      "\\delta_{s2}^2) + \\varepsilon} \\f$. "
      "Optionally, \\f$ \\delta_{\\mathrm{eff}} \\f$ is clamped to its historical maximum "
      "to enforce irreversible (monotonically increasing) damage.";

  options.set_parameter<TensorName<Scalar>>("fracture_energy");
  options.set("fracture_energy").doc() =
      "Critical fracture energy \\f$ G_c \\f$ (energy per unit area, e.g. N/mm). "
      "Together with \\f$ \\delta_0 \\f$ it sets the peak traction "
      "\\f$ T_{\\max} = G_c / (e\\,\\delta_0) \\f$.";

  options.set_parameter<TensorName<Scalar>>("softening_length_scale");
  options.set("softening_length_scale").doc() =
      "Characteristic softening length \\f$ \\delta_0 \\f$ (same units as the displacement jump). "
      "Smaller values produce faster exponential decay of the traction after peak.";

  options.set_parameter<TensorName<Scalar>>("tangential_weight");
  options.set("tangential_weight").doc() =
      "Dimensionless weight \\f$ \\beta \\f$ scaling the shear components in the effective "
      "displacement jump. \\f$ \\beta = 1 \\f$ gives equal weighting to normal and shear "
      "opening; \\f$ \\beta = 0 \\f$ makes the law purely mode-I.";

  options.set<double>("regularization_eps") = 1e-16;
  options.set("regularization_eps").doc() =
      "Small positive constant \\f$ \\varepsilon \\f$ added under the square root of "
      "\\f$ \\delta_{\\mathrm{eff}} \\f$ to prevent singular gradients when the displacement "
      "jump is exactly zero. Defaults to \\f$ 10^{-16} \\f$.";

  options.set<bool>("irreversible_damage") = false;
  options.set("irreversible_damage").doc() =
      "If true, damage is irreversible: the effective displacement jump driving the "
      "exponential softening is clamped to the historical maximum "
      "\\f$ \\max(\\delta_{\\mathrm{eff}},\\, \\delta_{\\mathrm{eff,max}}^{\\mathrm{old}}) \\f$, "
      "preventing traction recovery upon unloading. Defaults to false.";

  options.set_input("effective_displacement_jump_scalar_max_old") =
      VariableName(OLD_STATE, "effective_displacement_jump_scalar_max");
  options.set("effective_displacement_jump_scalar_max_old").doc() =
      "Historical maximum of \\f$ \\delta_{\\mathrm{eff}} \\f$ carried over from the previous "
      "time step. Only active when `irreversible_damage = true`.";

  options.set_output("effective_displacement_jump_scalar_max") =
      VariableName(STATE, "effective_displacement_jump_scalar_max");
  options.set("effective_displacement_jump_scalar_max").doc() =
      "Updated historical maximum of \\f$ \\delta_{\\mathrm{eff}} \\f$, written to state so "
      "it is available as `effective_displacement_jump_scalar_max_old` in the next step.";

  return options;
}

ExpTractionSeparation::ExpTractionSeparation(const OptionSet & options)
  : TractionSeparationModel(options),
    _Gc(declare_parameter<Scalar>("Gc", "fracture_energy")),
    _delta0(declare_parameter<Scalar>("delta0", "softening_length_scale")),
    _beta(declare_parameter<Scalar>("beta", "tangential_weight")),
    _eps(options.get<double>("regularization_eps")),
    _irreversible_damage(options.get<bool>("irreversible_damage")),
    _delta_eff_max_old(
        declare_input_variable<Scalar>("effective_displacement_jump_scalar_max_old")),
    _delta_eff_max(declare_output_variable<Scalar>("effective_displacement_jump_scalar_max"))
{
}

void
ExpTractionSeparation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto delta = _delta();
  const auto dn = delta(0);
  const auto ds1 = delta(1);
  const auto ds2 = delta(2);

  // Regularized effective displacement jump (raw, without irreversibility)
  const auto eps_scalar = Scalar::full(_eps, dn.options());
  const auto delta_eff_raw =
      neml2::sqrt(dn * dn + _beta * (ds1 * ds1 + ds2 * ds2) + eps_scalar);

  // Read historical maximum once (used for both out and dout_din paths)
  const Scalar old_max = _irreversible_damage ? _delta_eff_max_old() : Scalar();

  // Apply irreversibility: delta_eff = max(delta_eff_raw, old_max)
  const auto delta_eff =
      _irreversible_damage ? where(delta_eff_raw > old_max, delta_eff_raw, old_max) : delta_eff_raw;

  // Exponential damage factor: one_m_d = exp(-delta_eff / delta0)
  const auto one_m_d = neml2::exp(-delta_eff / _delta0);

  // Prefactor: c = Gc / delta0^2
  const auto c = _Gc / (_delta0 * _delta0);

  if (out)
  {
    _traction = Vec::fill(one_m_d * c * dn, one_m_d * c * ds1, one_m_d * c * ds2);
    _delta_eff_max = delta_eff;
  }

  if (dout_din)
  {
    if (_delta.is_dependent())
    {
      // Gradient of delta_eff w.r.t. delta components
      Scalar ddeff_ddn;
      Scalar ddeff_dds1;
      Scalar ddeff_dds2;

      if (_irreversible_damage)
      {
        const auto use_raw = delta_eff_raw > old_max;
        const auto zero = Scalar::zeros_like(dn);
        ddeff_ddn = where(use_raw, dn / delta_eff_raw, zero);
        ddeff_dds1 = where(use_raw, _beta * ds1 / delta_eff_raw, zero);
        ddeff_dds2 = where(use_raw, _beta * ds2 / delta_eff_raw, zero);
      }
      else
      {
        ddeff_ddn = dn / delta_eff_raw;
        ddeff_dds1 = _beta * ds1 / delta_eff_raw;
        ddeff_dds2 = _beta * ds2 / delta_eff_raw;
      }

      // Common factor: -(one_m_d / delta0)
      const auto factor = -(one_m_d / _delta0);

      // Prefactored diagonal term: one_m_d * c
      const auto oc = one_m_d * c;

      // Jacobian entries: dTi/ddeltaj = oc * delta_{ij} + c * delta_i * factor * ddeff_ddeltaj
      const auto dTn_ddn = oc + c * dn * factor * ddeff_ddn;
      const auto dTn_dds1 = c * dn * factor * ddeff_dds1;
      const auto dTn_dds2 = c * dn * factor * ddeff_dds2;

      const auto dTs1_ddn = c * ds1 * factor * ddeff_ddn;
      const auto dTs1_dds1 = oc + c * ds1 * factor * ddeff_dds1;
      const auto dTs1_dds2 = c * ds1 * factor * ddeff_dds2;

      const auto dTs2_ddn = c * ds2 * factor * ddeff_ddn;
      const auto dTs2_dds1 = c * ds2 * factor * ddeff_dds1;
      const auto dTs2_dds2 = oc + c * ds2 * factor * ddeff_dds2;

      _traction.d(_delta) = R2::fill(dTn_ddn,
                                     dTn_dds1,
                                     dTn_dds2,
                                     dTs1_ddn,
                                     dTs1_dds1,
                                     dTs1_dds2,
                                     dTs2_ddn,
                                     dTs2_dds1,
                                     dTs2_dds2);

      _delta_eff_max.d(_delta) = Vec::fill(ddeff_ddn, ddeff_dds1, ddeff_dds2);
    }

    if (_irreversible_damage && _delta_eff_max_old.is_dependent())
    {
      const auto use_raw = delta_eff_raw > old_max;
      const auto zero_s = Scalar::zeros_like(dn);
      const auto one_s = Scalar::ones_like(dn);
      const auto factor = -(one_m_d / _delta0);

      // d(delta_eff_max)/d(old_max): 0 when loading (raw wins), 1 when unloading (clamped)
      _delta_eff_max.d(_delta_eff_max_old) = where(use_raw, zero_s, one_s);

      // d(T_i)/d(old_max) = c * factor * d(delta_eff)/d(old_max) * delta_i
      const auto dT_factor = c * factor * where(use_raw, zero_s, one_s);
      _traction.d(_delta_eff_max_old) =
          Vec::fill(dT_factor * dn, dT_factor * ds1, dT_factor * ds2);
    }
  }
}
} // namespace neml2
