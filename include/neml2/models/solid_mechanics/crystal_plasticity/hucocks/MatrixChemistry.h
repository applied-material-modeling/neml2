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

#pragma once

#include "neml2/base/Registry.h"
#include "neml2/models/Data.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{

/// @brief Defines the geometry of a crystal system
/// This includes a basic definition of the crystal lattice,
/// via Bravais vectors and a CrystalClass object defining the
/// crystal symmetry as well as the definition of the geometry
/// of each slip system.
class MatrixChemistry : public Data
{
public:
  /// Input options
  static OptionSet expected_options();

  /// Setup from parameter set
  MatrixChemistry(const OptionSet & options);

  /// Get initial concentrations of some species
  Scalar initial_concentrations(std::vector<std::string> species) const;

  /// Get equilibrium concentrations of some species
  Scalar equilibrium_concentrations(std::vector<std::string> species) const;

private:
  /// @brief List of partipcipating species
  std::vector<std::string> _species;
  /// Initial concentrations of the species
  std::map<std::string, Real> _initial_concentrations;
  /// Equilibrium concentrations of the species
  std::map<std::string, Real> _equilibrium_concentrations;

  /// Make a scalar with the appropriate concentrations
  Scalar make_concentrations(const std::map<std::string, Real> & concentrations,
                             const std::vector<std::string> & species) const;
};

} // namespace neml2