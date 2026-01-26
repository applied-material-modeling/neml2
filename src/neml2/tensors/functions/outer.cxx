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

#include "neml2/tensors/functions/outer.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
#define DEFINE_OUTER(R, U)                                                                         \
  R outer(const U & u, const U & v)                                                                \
  {                                                                                                \
    return u.base_unsqueeze(-1, v.base_dim()) * v.base_unsqueeze(0, u.base_dim());                 \
  }                                                                                                \
                                                                                                   \
  R outer(const U & u) { return outer(U(u), U(u)); }                                               \
  static_assert(true)

DEFINE_OUTER(Tensor, Tensor);
DEFINE_OUTER(R2, Vec);
DEFINE_OUTER(R4, R2);
DEFINE_OUTER(SSR4, SR2);
} // namespace neml2
