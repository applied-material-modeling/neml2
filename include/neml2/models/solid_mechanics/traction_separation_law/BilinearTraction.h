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
 * @brief Bilinear cohesive-zone traction with internal damage state.
 *
 * Given the effective separation \f$ \delta_m \f$, the critical (damage-onset)
 * and full (failure) separations \f$ \delta_c \f$ and \f$ \delta_f \f$, and the
 * per-component displacement-jump pieces, this model:
 *
 *   1. Computes the bilinear damage variable
 *      \f[
 *         d_\text{trial} = \begin{cases}
 *           0 & \delta_m \le \delta_c \\
 *           \dfrac{\delta_f (\delta_m - \delta_c)}
 *                 {\delta_m (\delta_f - \delta_c)}
 *             & \delta_c < \delta_m < \delta_f \\
 *           1 & \delta_m \ge \delta_f
 *         \end{cases}
 *      \f]
 *   2. Caps it for irreversibility: \f$ d = \max(d_\text{trial}, d_{n-1}) \f$
 *      where \f$ d_{n-1} \f$ is the previous-step damage (auto-declared via
 *      `history_name` on the `damage` output).
 *   3. Assembles the traction:
 *      \f[
 *         T_n = K(1-d)\delta_n^+ + K \delta_n^-, \quad
 *         T_{si} = K(1-d)\delta_{si}.
 *      \f]
 *
 * `critical_separation` and `full_separation` are declared with
 * `allow_nonlinear=true` so the user can supply them either as plain numeric
 * literals or wire them to upstream models (e.g.
 * `CamanhoDavilaCriticalSeparation`, `BenzeggaghKenaneFullSeparation`).
 *
 * The damage variable is intentionally an internal computational artifact of
 * the bilinear law — it is exposed as a secondary `damage` output for
 * inspection but the framework's history-variable mechanism (and the cap in
 * step 2) handles irreversibility internally; no external `IrreversibleScalar`
 * is needed.
 */
class BilinearTraction : public Model
{
public:
  static OptionSet expected_options();

  BilinearTraction(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Traction Vec
  Variable<Vec> & _to;

  /// Damage scalar (current step, irreversibility-capped)
  Variable<Scalar> & _d;

  /// Damage scalar (previous step, auto-declared via `history_name`)
  const Variable<Scalar> & _d_old;

  /// Effective separation \f$ \delta_m \f$
  const Variable<Scalar> & _delta_m;

  /// Critical (damage-onset) separation \f$ \delta_c \f$ (nonlinear-capable parameter)
  const Scalar & _delta_init;

  /// Full (failure) separation \f$ \delta_f \f$ (nonlinear-capable parameter)
  const Scalar & _delta_final;

  /// Normal separation \f$ \delta_n^\text{sep} \f$ (typically Macaulay-positive)
  const Variable<Scalar> & _dn_sep;

  /// Optional normal penetration \f$ \delta_n^\text{pen} \f$ (typically Macaulay-negative);
  /// nullptr if the user did not supply it
  const Variable<Scalar> * _dn_pen;

  /// Tangential separations
  const Variable<Scalar> & _ds1;
  const Variable<Scalar> & _ds2;

  /// Penalty stiffness K
  const Scalar & _K;
};
} // namespace neml2
