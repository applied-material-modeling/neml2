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

#include "neml2/models/phase_field_fracture/StrainEnergyDensity.h"
#include "neml2/base/EnumSelection.h"
#include "neml2/tensors/Scalar.h"
#include <string>

namespace neml2
{
OptionSet
StrainEnergyDensity::expected_options()
{
  OptionSet options = Model::expected_options();

  options.set_input("strain") = VariableName(STATE, "internal", "Ee");
  options.set("strain").doc() = "Elastic strain";

  options.set_output("strain_energy_density_active") = VariableName(STATE, "psie_active");
  options.set("strain_energy_density_active").doc() = "Active part of the strain energy density";

  options.set_output("strain_energy_density_inactive") = VariableName(STATE, "psie_inactive");
  options.set("strain_energy_density_inactive").doc() =
      "Inactive part of the strain energy density";

  EnumSelection type_selection({"NONE", "SPECTRAL", "VOLDEV"}, "NONE");
  options.set<EnumSelection>("decomposition") = type_selection;
  options.set("decomposition").doc() =
      "Strain energy density decomposition types, options are: " + type_selection.candidates_str();
  return options;
}

StrainEnergyDensity::StrainEnergyDensity(const OptionSet & options)
  : Model(options),
    _strain(declare_input_variable<SR2>("strain")),
    _psie_active(declare_output_variable<Scalar>("strain_energy_density_active")),
    _psie_inactive(declare_output_variable<Scalar>("strain_energy_density_inactive")),
    _decomposition(options.get<EnumSelection>("decomposition").as<DecompositionType>())
{
}
} // namespace neml2
