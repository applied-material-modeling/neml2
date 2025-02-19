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

#include "neml2/models/solid_mechanics/crystal_plasticity/hucocks/CurrentConcentrations.h"
#include "neml2/models/solid_mechanics/crystal_plasticity/hucocks/MatrixChemistry.h"
#include "neml2/models/solid_mechanics/crystal_plasticity/hucocks/Precipitate.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/list_tensors.h"
#include "neml2/tensors/functions/sum.h"

namespace neml2
{
register_NEML2_object(CurrentConcentrations);

OptionSet
CurrentConcentrations::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() = "Calculates the current mass fraction concentration in the matrix"
                  " as \\f$X\\f$ ";

  options.set_output("current_concentrations") =
      VariableName(STATE, "internal", "current_concentrations");
  options.set("current_concentrations").doc() = "Output: current checmical composition of matrix";

  options.set_input("volume_fractions") = VariableName(STATE, "internal", "volume_fractions");
  options.set("volume_fractions").doc() = "Input: current volume fractions of precipitates";

  options.set<std::string>("matrix_chemistry_name") = "matrix_chemistry";
  options.set("matrix_chemistry_name").doc() =
      "The name of the Data object containing the matrix chemistry";

  options.set<std::vector<std::string>>("precipitate_names") = std::vector<std::string>();
  options.set("precipitate_names").doc() = "The names of the precipitates in the system";

  return options;
}

CurrentConcentrations::CurrentConcentrations(const OptionSet & options)
  : Model(options),
    _matrix_chemistry(
        register_data<MatrixChemistry>(options.get<std::string>("matrix_chemistry_name"))),
    _precipitates(
        register_data<Precipitate>(options.get<std::vector<std::string>>("precipitate_names"))),
    _c(declare_output_variable<Scalar>("current_concentrations", _matrix_chemistry.nspecies())),
    _f(declare_input_variable<Scalar>("volume_fractions", _precipitates.size()))
{
}

void
CurrentConcentrations::set_value(bool out, bool /*dout_din*/, bool /*d2out_din2*/)
{
  if (out)
  {
    Scalar f_total = 1.0 - batch_sum(_f.value(), -1);
    _c = _matrix_chemistry.initial_concentrations(_matrix_chemistry.species());
    for (size_t i = 0; i < _precipitates.size(); i++)
      _c = _c - _precipitates[i]->concentrations(_matrix_chemistry.species()) *
                    _f.value().batch_index({Size(i)});
    _c = _c / f_total;
  }
}
} // namespace neml2
