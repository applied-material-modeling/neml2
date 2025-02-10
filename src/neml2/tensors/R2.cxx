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

#include "neml2/tensors/R2.h"

#include "neml2/tensors/Rot.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SSR4.h"

#include "neml2/tensors/mandel_notation.h"

namespace neml2
{

R2::R2(const SR2 & S)
  : R2(mandel_to_full(S))
{
}

R2::R2(const WR2 & W)
  : R2(skew_to_full(W))
{
}

R2::R2(const Rot & r)
  : R2(r.euler_rodrigues())
{
}

R4
R2::identity_map(const TensorOptions & options)
{
  return R4(at::eye(9, options).view({3, 3, 3, 3}));
}
} // namespace neml2
