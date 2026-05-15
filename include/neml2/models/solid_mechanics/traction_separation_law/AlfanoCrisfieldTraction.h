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

#include "neml2/models/solid_mechanics/traction_separation_law/BilinearMixedModeTraction.h"

namespace neml2
{
/**
 * @brief Bilinear mixed-mode cohesive-zone law with the power-law propagation criterion
 *        (Alfano & Crisfield 2001).
 *
 * Specializes BilinearMixedModeTraction with the power-law formula for the full-degradation
 * jump in the opening branch. The mixed-mode total fracture toughness is determined by
 * \f[
 *   \left(\frac{G_I}{G_{Ic}}\right)^\eta + \left(\frac{G_{II}}{G_{IIc}}\right)^\eta = 1,
 * \f]
 * which under the bilinear envelope yields
 * \f[
 *   \delta_\text{final} = \frac{2(1+\beta^2)}{K\,\delta_\text{init}}
 *     \left[\left(\frac{1}{G_{Ic}}\right)^\eta
 *         + \left(\frac{\beta^2}{G_{IIc}}\right)^\eta\right]^{-1/\eta}.
 * \f]
 * The exponent \f$ \eta \f$ is the power-law shape parameter (commonly 1 or 2).
 *
 * Reference: Alfano, G. & Crisfield, M.A. (2001). "Finite element interface models for the
 * delamination analysis of laminated composites: mechanical and computational issues."
 * International Journal for Numerical Methods in Engineering 50, 1701–1736.
 */
class AlfanoCrisfieldTraction : public BilinearMixedModeTraction
{
public:
  static OptionSet expected_options();

  AlfanoCrisfieldTraction(const OptionSet & options);

protected:
  DeltaFinalResult compute_delta_final(const DeltaFinalContext & ctx, bool dout_din) const override;
};
} // namespace neml2
