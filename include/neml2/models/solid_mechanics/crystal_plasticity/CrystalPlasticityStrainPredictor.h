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
class SR2;

/**
 * @brief Warm-up predictor for crystal plasticity models.
 *
 * Computes an initial guess for the elastic strain as
 * \f$ \varepsilon^e = \Delta t \cdot d \cdot s \f$
 * where \f$ \Delta t = t - t_n \f$ is the time increment, \f$ d \f$ is the deformation rate,
 * and \f$ s \f$ is a user-supplied scale factor.
 */
class CrystalPlasticityStrainPredictor : public Model
{
public:
  static OptionSet expected_options();

  CrystalPlasticityStrainPredictor(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Deformation rate
  const Variable<SR2> & _D;
  /// Current time
  const Variable<Scalar> & _t;
  /// Previous time (t~1)
  const Variable<Scalar> & _tn;
  /// Scale factor
  const Scalar & _scale;
  /// Elastic strain output (initial guess)
  Variable<SR2> & _Ee;
};
} // namespace neml2
