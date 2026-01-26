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

namespace neml2
{
class Tensor;
class Rot;
class R2;
class SR2;
class R4;
class SSR4;

/// Inverse of a second order or fourth order tensor
Tensor inv(const Tensor &);

/// Inverse of a rotation (MRP)
Rot inv(const Rot &);

///@{
/// Inverse of a second order tensor
R2 inv(const R2 &);
SR2 inv(const SR2 &);
///@}

///@{
/// Inverse of a fourth order tensor, i.e., \f$A^{-1}_{ijmn} A_{mnkl} = \delta_{il} \delta_{jk}\f$
R4 inv(const R4 &);
SSR4 inv(const SSR4 &);
///@}
} // namespace neml2
