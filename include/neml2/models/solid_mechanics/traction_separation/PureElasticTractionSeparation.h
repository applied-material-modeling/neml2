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

#include "neml2/models/solid_mechanics/traction_separation/TractionSeparation.h"

namespace neml2
{
class Scalar;

/**
 * @brief Purely linear-elastic traction-separation law.
 *
 * Maps the displacement jump to traction through a constant orthotropic stiffness:
 * \f[
 *   \boldsymbol{T} = \begin{bmatrix} K_n & 0 & 0 \\ 0 & K_t & 0 \\ 0 & 0 & K_t \end{bmatrix}
 *                    \boldsymbol{\delta},
 * \f]
 * where \f$ K_n \f$ is the normal stiffness and \f$ K_t \f$ is the tangential stiffness (assumed
 * isotropic in the two tangential directions). The Jacobian is a constant diagonal matrix.
 *
 * The model treats opening (\f$ \delta_n > 0 \f$) and closing (\f$ \delta_n < 0 \f$) symmetrically;
 * if interface compression must be penalized differently, pair this law with a separate contact
 * penalty term.
 */
class PureElasticTractionSeparation : public TractionSeparation
{
public:
  static OptionSet expected_options();

  PureElasticTractionSeparation(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Elastic stiffness in the normal direction
  const Scalar & _Kn;

  /// Elastic stiffness in both tangential directions
  const Scalar & _Kt;
};
} // namespace neml2
