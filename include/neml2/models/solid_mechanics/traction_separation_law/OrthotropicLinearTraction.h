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
class Vec;

/**
 * @brief Orthotropic linear-elastic interface traction.
 *
 * \f[
 *   T_n = K_n \delta_n^\text{sep} + [K_\text{pen} \delta_n^\text{pen}], \quad
 *   T_{si} = K_t \delta_{si}.
 * \f]
 *
 * The bracketed interpenetration term is included only if the user supplies
 * the optional `normal_penetration` input (typically the Macaulay-negative
 * part of the normal jump produced by `MacaulaySplit`) along with the
 * required `penalty_stiffness` parameter. When omitted, the model produces
 * zero normal traction under interpenetration; the user opts in to elastic
 * interpenetration resistance by wiring both. The penalty stiffness is
 * declared separately from the elastic normal stiffness so the user can
 * choose a stiffer contact penalty than the bonded normal stiffness.
 */
class OrthotropicLinearTraction : public Model
{
public:
  static OptionSet expected_options();

  OrthotropicLinearTraction(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Traction Vec (assembled internally via Vec::fill)
  Variable<Vec> & _to;

  /// Normal separation \f$ \delta_n^\text{sep} \f$ (typically Macaulay-positive)
  const Variable<Scalar> & _dn_sep;

  /// Optional normal penetration \f$ \delta_n^\text{pen} \f$ (typically Macaulay-negative);
  /// nullptr if the user did not supply it
  const Variable<Scalar> * _dn_pen;

  const Variable<Scalar> & _ds1;
  const Variable<Scalar> & _ds2;

  const Scalar & _Kn;
  const Scalar & _Kt;

  /// Optional penalty stiffness for interpenetration; nullptr if `normal_penetration` was not
  /// supplied
  const Scalar * _Kpen;
};
} // namespace neml2
