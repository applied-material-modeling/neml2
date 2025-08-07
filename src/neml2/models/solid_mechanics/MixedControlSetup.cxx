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

#include "neml2/models/solid_mechanics/MixedControlSetup.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/diagonalize.h"

namespace neml2
{
register_NEML2_object(MixedControlSetup);

OptionSet
MixedControlSetup::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Object to setup a model for mixed stress/strain control.  Copies the values of the "
      "fixed_values (the input strain or stress) and the mixed_state (the conjugate stress or "
      "strain values) into the stress and strain tensors used by the model.";

  options.set_input("control") = VariableName(FORCES, "control");
  options.set("control").doc() =
      "The name of the control signal.  Values less than the threshold are "
      "strain control, greater are stress control";

  options.set<TensorName<SR2>>("threshold") = TensorName<SR2>("0.5");
  options.set("threshold").doc() = "The threshold to switch between strain and stress control";

  options.set_input("mixed_state") = VariableName(STATE, "mixed_state");
  options.set("mixed_state").doc() = "The name of the mixed state tensor. This holds the conjugate "
                                     "values to those being controlled";

  options.set_input("fixed_values") = VariableName(FORCES, "fixed_values");
  options.set("fixed_values").doc() = "The name of the fixed values, i.e. the actual strain or "
                                      "stress values being imposed on the model";

  options.set_output("cauchy_stress") = VariableName(STATE, "S");
  options.set("cauchy_stress").doc() = "The name of the Cauchy stress tensor";

  options.set_output("strain") = VariableName(STATE, "E");
  options.set("strain").doc() = "The name of the strain tensor";

  return options;
}

MixedControlSetup::MixedControlSetup(const OptionSet & options)
  : Model(options),
    _threshold(declare_buffer<SR2>("c", "threshold")),
    _control(declare_input_variable<SR2>("control")),
    _fixed_values(declare_input_variable<SR2>("fixed_values")),
    _mixed_state(declare_input_variable<SR2>("mixed_state")),
    _stress(declare_output_variable<SR2>("cauchy_stress")),
    _strain(declare_output_variable<SR2>("strain"))
{
}

void
MixedControlSetup::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto [dstrain, dstress] = make_operators(_control());

  if (out)
  {
    // Even benign in place operations get errors
    _stress = dstress * _fixed_values + dstrain * _mixed_state;
    _strain = dstrain * _fixed_values + dstress * _mixed_state;
  }

  if (dout_din)
  {
    if (_fixed_values.is_dependent())
    {
      _stress.d(_fixed_values) = dstress;
      _strain.d(_fixed_values) = dstrain;
    }

    if (_mixed_state.is_dependent())
    {
      _stress.d(_mixed_state) = dstrain;
      _strain.d(_mixed_state) = dstress;
    }
  }

  // All zero
  (void)(d2out_din2);
}

std::pair<SSR4, SSR4>
MixedControlSetup::make_operators(const SR2 & control) const
{
  auto strain_select = control <= _threshold;
  auto stress_select = control > _threshold;

  // This also converts these to floats
  auto ones_stress = Tensor(strain_select.to(control.options()), control.batch_sizes());
  auto ones_strain = Tensor::ones_like(control) - ones_stress;

  auto dstrain = base_diagonalize(ones_stress);
  auto dstress = base_diagonalize(ones_strain);

  return {dstrain, dstress};
}

} // namespace neml2
