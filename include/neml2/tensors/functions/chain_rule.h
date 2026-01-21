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

#include "neml2/tensors/Derivative.h"

namespace neml2
{

/// Apply chain rule on two first-order derivatives
Derivative<1> chain_rule(const Derivative<1> & dy_du, const Derivative<1> & du_dx);

/// Apply second order chain rule on d2y_du1u2, du1_dx1, du2_dx2
/// In einstein notation: ipq, pj, qk -> ijk
Derivative<2> chain_rule(const Derivative<2> & d2y_du1u2,
                         const Derivative<1> * du_dx1,
                         const Derivative<1> * du_dx2);

/// Apply second order chain rule on dy_du, d2u_dx1x2
/// In einstein notation: ip, pjk -> ijk
Derivative<2> chain_rule(const Derivative<1> & dy_du, const Derivative<2> & d2u_dx1x2);

} // namespace neml2
