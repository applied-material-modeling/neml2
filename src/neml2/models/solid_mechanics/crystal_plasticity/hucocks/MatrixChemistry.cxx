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

#include "neml2/models/solid_mechanics/crystal_plasticity/hucocks/MatrixChemistry.h"

namespace neml2
{

register_NEML2_object(MatrixChemistry);

OptionSet
MatrixChemistry::expected_options()
{
  OptionSet options = Data::expected_options();

  options.doc() =
      "A Data object storing the composition of the matrix phase.  This includes the available "
      "species, their initial conceentrations, and their equilibrium concentrations.";

  options.set<std::vector<std::string>>("species");
  options.set("species").doc() = "The participating chemical species";

  options.set<std::vector<Real>>("initial_concentrations");
  options.set("initial_concentrations").doc() =
      "The initial concentrations of the species.  Must be the same size as species";

  options.set<std::vector<Real>>("equilibrium_concentrations");
  options.set("equilibrium_concentrations").doc() =
      "The equilibrium concentrations of the species.  Must be the same size as species";

  return options;
}

MatrixChemistry::MatrixChemistry(const OptionSet & options)
  : Data(options),
    _species(options.get<std::vector<std::string>>("species"))
{
  auto initial_concentrations = options.get<std::vector<Real>>("initial_concentrations");
  auto equilibrium_concentrations = options.get<std::vector<Real>>("equilibrium_concentrations");
  neml_assert(initial_concentrations.size() == _species.size(),
              "Initial concentrations must be the same size as species");
  neml_assert(equilibrium_concentrations.size() == _species.size(),
              "Equilibrium concentrations must be the same size as species");
  for (size_t i = 0; i < _species.size(); i++)
  {
    _initial_concentrations[_species[i]] = initial_concentrations[i];
    _equilibrium_concentrations[_species[i]] = equilibrium_concentrations[i];
  }
}

Scalar
MatrixChemistry::initial_concentrations(std::vector<std::string> species) const
{
  return make_concentrations(_initial_concentrations, species);
}

Scalar
MatrixChemistry::equilibrium_concentrations(std::vector<std::string> species) const
{
  return make_concentrations(_equilibrium_concentrations, species);
}

Scalar
MatrixChemistry::make_concentrations(const std::map<std::string, Real> & concentrations,
                                     const std::vector<std::string> & species) const
{
  std::vector<Real> result(species.size(), 0.0);
  for (size_t i = 0; i < species.size(); i++)
  {
    auto s = species[i];
    auto it = concentrations.find(s);
    neml_assert(it != concentrations.end(), "Species " + s + " not found in concentrations");
    result[i] = it->second;
  }
  return Scalar::create(result);
}

} // namespace neml2