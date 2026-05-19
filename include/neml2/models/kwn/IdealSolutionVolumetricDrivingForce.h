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

#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

/// Ideal-solution volumetric Gibbs free energy of precipitation.
///
/// Computes the molar Gibbs free energy of precipitation under the
/// ideal-solution approximation,
/// \f$ \Delta g = R T \sum_k w_k \ln(c_k / c_k^{\mathrm{eq}}) \f$,
/// where the sum runs over the species participating in the precipitate and
/// \f$ w_k \f$ are user-supplied stoichiometric weights (default 1, matching
/// the Hu--Cocks "product of all components" convention for compound
/// precipitates).
///
/// The output is per mole of precipitate, with units J/mol, so it can be fed
/// directly to `NucleationBarrierAndCriticalRadius` (and similar objects) that
/// adopt the molar convention throughout the KWN module.
class IdealSolutionVolumetricDrivingForce : public Model
{
public:
  static OptionSet expected_options();

  IdealSolutionVolumetricDrivingForce(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Temperature
  const Variable<Scalar> & _T;

  /// Current matrix concentrations per species
  std::vector<const Variable<Scalar> *> _xs;

  /// Equilibrium matrix concentrations per species
  std::vector<const Scalar *> _x_eqs;

  /// Stoichiometric weights per species
  std::vector<const Scalar *> _ws;

  /// Gas constant
  const Scalar & _R_g;

  /// Molar Gibbs free energy of precipitation
  Variable<Scalar> & _dg;
};
} // namespace neml2
