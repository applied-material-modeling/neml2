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

#include "neml2/models/solid_mechanics/LinearIsotropicElasticJ2TrialStressUpdate.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(LinearIsotropicElasticJ2TrialStressUpdate);

OptionSet
LinearIsotropicElasticJ2TrialStressUpdate::expected_options()
{
  OptionSet options = ElasticityInterface<Model, 2>::expected_options();

  options.doc() = "Update the trial stress under the assumptions of J2 plasticity and isotropic "
                  "linear elasticity";

  options.set_input("elastic_trial_stress") = VariableName(FORCES, "s");
  options.set("elastic_trial_stress").doc() = "Initial trial stress assuming a purely elastic step";

  options.set_input("equivalent_plastic_strain") = VariableName(STATE, "ep");
  options.set("equivalent_plastic_strain").doc() =
      "Current guess for the equivalent plastic strain";

  options.set_output("updated_trial_stress") = VariableName(STATE, "s");
  options.set("updated_trial_stress").doc() =
      "Trial stress corrected for the current increment of plastic deformation";

  return options;
}

LinearIsotropicElasticJ2TrialStressUpdate::LinearIsotropicElasticJ2TrialStressUpdate(
    const OptionSet & options)
  : ElasticityInterface<Model, 2>(options),
    _elastic_trial_stress(declare_input_variable<Scalar>("elastic_trial_stress")),
    _inelastic_strain(declare_input_variable<Scalar>("equivalent_plastic_strain")),
    _inelastic_strain_old(declare_input_variable<Scalar>(_inelastic_strain.name().old())),
    _updated_trial_stress(declare_output_variable<Scalar>("updated_trial_stress")),
    _converter(_constant_types, _need_derivs)
{
}

void
LinearIsotropicElasticJ2TrialStressUpdate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const auto [G, dG] = _converter.convert(_constants, ElasticConstant::SHEAR_MODULUS);
  const auto three_shear = 3.0 * G;

  if (out)
    _updated_trial_stress =
        _elastic_trial_stress - three_shear * (_inelastic_strain - _inelastic_strain_old);

  if (dout_din)
  {
    if (_elastic_trial_stress.is_dependent())
      _updated_trial_stress.d(_elastic_trial_stress) =
          imap<Scalar>(_elastic_trial_stress.options()).as<Scalar>();

    if (_inelastic_strain.is_dependent())
      _updated_trial_stress.d(_inelastic_strain) = -three_shear;

    if (_inelastic_strain_old.is_dependent())
      _updated_trial_stress.d(_inelastic_strain_old) = three_shear;

    if (const auto * const p1 = nl_param(neml2::name(_constant_types[0])))
      _updated_trial_stress.d(*p1) = -3.0 * dG[0] * (_inelastic_strain - _inelastic_strain_old);

    if (const auto * const p2 = nl_param(neml2::name(_constant_types[1])))
      _updated_trial_stress.d(*p2) = -3.0 * dG[1] * (_inelastic_strain - _inelastic_strain_old);
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
