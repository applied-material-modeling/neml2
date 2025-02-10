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

#include "neml2/tensors/R2Base.h"

namespace neml2
{
class Rot;
class SR2;
class WR2;
class R4;

/**
 * @brief Second order tensor without symmetry.
 *
 * The storage space is (3,3).
 */
class R2 : public R2Base<R2>
{
public:
  using R2Base<R2>::R2Base;

  /// @brief Form a full R2 from a symmetric tensor
  /// @param S Mandel-convention symmetric tensor
  R2(const SR2 & S);

  /// @brief Form a full R2 from a skew-symmetric tensor
  /// @param W skew-vector convention skew-symmetric tensor
  R2(const WR2 & W);

  /// @brief Form rotation matrix from vector
  /// @param r rotation vector
  explicit R2(const Rot & r);

  /// The derivative of a R2 with respect to itself
  [[nodiscard]] static R4 identity_map(const TensorOptions & options = default_tensor_options());
};

} // namespace neml2
