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

#include "neml2/models/solid_mechanics/crystal_plasticity/hucocks/Precipitate.h"

namespace neml2
{

register_NEML2_object(Precipitate);

OptionSet
Precipitate::expected_options()
{
  OptionSet options = Species::expected_options();

  options.doc() = "A Data object storing the composition of a precipitate.";

  options.set<std::vector<Real>>("concentrations");
  options.set("concentrations").doc() =
      "The chemical composition of the precipitate.  Must be the same size as species";

  return options;
}

Precipitate::Precipitate(const OptionSet & options)
  : Species(options)
{
  auto concentrations = options.get<std::vector<Real>>("concentrations");
  neml_assert(concentrations.size() == _species.size(),
              "Concentrations must be the same size as species");
  for (size_t i = 0; i < _species.size(); i++)
    _concentrations[_species[i]] = concentrations[i];
}

Scalar
Precipitate::concentrations(const std::vector<std::string> & species) const
{
  return make_concentrations(_concentrations, species);
}
} // namespace neml2