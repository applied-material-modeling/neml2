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
 * @brief Bilinear mixed-mode cohesive-zone law with the Benzeggagh-Kenane (BK) propagation
 *        criterion (Camanho & Davila 2002).
 *
 * Specializes BilinearMixedModeTraction with the BK formula for the full-degradation jump in
 * the opening branch:
 * \f[
 *   \delta_\text{final} = \frac{2}{K\,\delta_\text{init}} \left(
 *     G_{Ic} + (G_{IIc} - G_{Ic}) \left(\frac{\beta^2}{1+\beta^2}\right)^\eta
 *   \right).
 * \f]
 * The exponent \f$ \eta \f$ is the BK material parameter (typically 1.5–2.5 for laminated
 * composites).
 *
 * Reference: Camanho, P.P. & Davila, C.G. (2002). "Mixed-Mode Decohesion Finite Elements for
 * the Simulation of Delamination in Composite Materials." NASA TM-2002-211737.
 */
class CamanhoDavilaTraction : public BilinearMixedModeTraction
{
public:
  static OptionSet expected_options();

  CamanhoDavilaTraction(const OptionSet & options);

protected:
  DeltaFinalResult compute_delta_final(const DeltaFinalContext & ctx, bool dout_din) const override;
};
} // namespace neml2
