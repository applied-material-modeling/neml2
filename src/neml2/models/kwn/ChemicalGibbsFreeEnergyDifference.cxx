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

#include "neml2/models/kwn/ChemicalGibbsFreeEnergyDifference.h"

#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(ChemicalGibbsFreeEnergyDifference);

OptionSet
ChemicalGibbsFreeEnergyDifference::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the chemical Gibbs free energy difference.";

  options.set<std::vector<VariableName>>("concentration_differences");
  options.set("concentration_differences").doc() =
      "Concentration differences for each species";

  options.set_parameter<std::vector<TensorName<Scalar>>>("chemical_potentials");
  options.set("chemical_potentials").doc() = "Chemical potentials in the matrix";

  options.set_parameter<std::vector<TensorName<Scalar>>>("equilibrium_potentials");
  options.set("equilibrium_potentials").doc() = "Equilibrium chemical potentials";

  options.set_output("chemical_gibbs_free_energy");
  options.set("chemical_gibbs_free_energy").doc() = "Chemical Gibbs free energy difference";

  return options;
}

ChemicalGibbsFreeEnergyDifference::ChemicalGibbsFreeEnergyDifference(const OptionSet & options)
  : Model(options),
    _dg(declare_output_variable<Scalar>("chemical_gibbs_free_energy"))
{
  for (const auto & v : options.get<std::vector<VariableName>>("concentration_differences"))
    _dxs.push_back(&declare_input_variable<Scalar>(v));

  const auto mu_refs = options.get<std::vector<TensorName<Scalar>>>("chemical_potentials");
  const auto mu_eq_refs = options.get<std::vector<TensorName<Scalar>>>("equilibrium_potentials");

  neml_assert(mu_refs.size() == _dxs.size(),
              "Number of chemical potentials (",
              mu_refs.size(),
              ") does not match number of concentration differences (",
              _dxs.size(),
              ").");

  neml_assert(mu_eq_refs.size() == _dxs.size(),
              "Number of equilibrium potentials (",
              mu_eq_refs.size(),
              ") does not match number of concentration differences (",
              _dxs.size(),
              ").");

  _mus.resize(_dxs.size());
  _mu_eqs.resize(_dxs.size());
  for (std::size_t i = 0; i < _dxs.size(); i++)
  {
    _mus[i] = &declare_parameter<Scalar>(
        "mu_" + std::to_string(i), mu_refs[i], /*allow_nonlinear=*/true);
    _mu_eqs[i] = &declare_parameter<Scalar>(
        "mu_eq_" + std::to_string(i), mu_eq_refs[i], /*allow_nonlinear=*/true);
  }
}

void
ChemicalGibbsFreeEnergyDifference::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto sum = Scalar::zeros_like((*_dxs[0])());

  for (std::size_t i = 0; i < _dxs.size(); i++)
    sum = sum + (*_dxs[i])() * (*_mus[i] - *_mu_eqs[i]);

  if (out)
    _dg = sum;

  if (dout_din)
  {
    for (std::size_t i = 0; i < _dxs.size(); i++)
    {
      const auto coef = (*_mus[i] - *_mu_eqs[i]);

      if (_dxs[i]->is_dependent())
        _dg.d(*_dxs[i]) = coef;

      if (const auto * const mu = nl_param("mu_" + std::to_string(i)))
        _dg.d(*mu) += (*_dxs[i])();

      if (const auto * const mu_eq = nl_param("mu_eq_" + std::to_string(i)))
        _dg.d(*mu_eq) += -(*_dxs[i])();
    }
  }
}
} // namespace neml2
