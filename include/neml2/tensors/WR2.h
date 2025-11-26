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

#include "neml2/tensors/VecBase.h"

namespace neml2
{
class Scalar;
class R2;
class Rot;
class SR2;
class WSR4;

/**
 * @brief A skew-symmetric second order tensor, represented as an axial vector
 *
 * The storage space is (3).
 */
class WR2 : public VecBase<WR2>
{
public:
  using VecBase<WR2>::VecBase;

  /// Skew-symmetrize a R2 then fill
  WR2(const R2 & T);

  /// Accessor
  Scalar operator()(Size i, Size j) const;

  /// Exponential map to make this into a rotation (Rot)
  Rot exp_map() const;

  /// Derivative of the exponential map
  R2 dexp_map() const;
};

} // namespace neml2
