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
 * @brief 3D coupled exponential cohesive law of Salehani & Irani with internal damage state.
 *
 * \f[
 *   T_i = a_i b_i (1 - d), \quad
 *   x = b_n + b_{s1}^2 + b_{s2}^2,
 * \f]
 * where \f$ b_n = \delta_n^\text{sep} / \delta_{u0,n} \f$,
 * \f$ b_{si} = \delta_{si} / (\sqrt{2}\,\delta_{u0,t}) \f$,
 * \f$ a_n = e\, T_n^{\max} \f$, and \f$ a_t = \sqrt{2e}\, T_t^{\max} \f$.
 * The damage variable is computed as
 * \f[
 *   d_\text{trial} = 1 - \exp(-x), \quad d = \max(d_\text{trial}, d_{n-1}),
 * \f]
 * with \f$ d_{n-1} \f$ the previous-step damage (auto-declared via
 * `history_name` on the `damage` output). For monotonically increasing
 * \f$ x \f$ the formulation reduces exactly to the original
 * \f$ T_i = a_i b_i \exp(-x) \f$; for load-unload-reload schedules the
 * damage cap freezes the softness at its historical peak, preventing the
 * un-physical healing that the bare exponential law would otherwise produce.
 *
 * The internal tangential characteristic length is \f$ \sqrt{2}\,\delta_{u0,t} \f$
 * to match the original publication's convention.
 *
 * The cohesive law uses only the *separation* part of the normal jump
 * (\f$ \delta_n^\text{sep} \f$), so it produces zero normal traction under
 * interpenetration. To resist interpenetration, supply the optional
 * `normal_penetration` input along with the required `penalty_stiffness`
 * parameter; the model then adds \f$ K_\text{pen}\,\delta_n^\text{pen} \f$
 * to \f$ T_n \f$.
 */
class SalehaniIraniTraction : public Model
{
public:
  static OptionSet expected_options();

  SalehaniIraniTraction(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Traction Vec
  Variable<Vec> & _to;

  /// Damage scalar (current step, irreversibility-capped)
  Variable<Scalar> & _d;

  /// Damage scalar (previous step, auto-declared via `history_name`)
  const Variable<Scalar> & _d_old;

  /// Normal separation \f$ \delta_n^\text{sep} \f$ (typically Macaulay-positive)
  const Variable<Scalar> & _dn_sep;

  /// Optional normal penetration \f$ \delta_n^\text{pen} \f$; nullptr if the user did not
  /// supply it
  const Variable<Scalar> * _dn_pen;

  const Variable<Scalar> & _ds1;
  const Variable<Scalar> & _ds2;

  /// Normal characteristic length (raw user input)
  const Scalar & _delta_u0_n;

  /// Tangential characteristic length (raw user input; sqrt(2) is applied internally)
  const Scalar & _delta_u0_t;

  /// Maximum normal traction
  const Scalar & _Tmax_n;

  /// Maximum shear traction
  const Scalar & _Tmax_t;

  /// Optional penalty stiffness for interpenetration; nullptr if `normal_penetration` was not
  /// supplied (in which case `penalty_stiffness` is also not declared)
  const Scalar * _Kpen;
};
} // namespace neml2
