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

#include "neml2/tensors/functions/diag_embed.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
#define DEFINE_BATCH_DIAG_EMDED(T)                                                                 \
  T batch_diag_embed(const T & a, Size offset, Size d1, Size d2)                                   \
  {                                                                                                \
    return T(at::diag_embed(                                                                       \
                 a, offset, d1 < 0 ? d1 - a.base_dim() : d1, d2 < 0 ? d2 - a.base_dim() : d2),     \
             a.batch_dim() + 1);                                                                   \
  }                                                                                                \
  static_assert(true)
FOR_ALL_TENSORBASE(DEFINE_BATCH_DIAG_EMDED);

Tensor
base_diag_embed(const Tensor & a, Size offset, Size d1, Size d2)
{
  return Tensor(
      at::diag_embed(
          a, offset, d1 < 0 ? d1 : d1 + a.batch_dim() + 1, d2 < 0 ? d2 : d2 + a.batch_dim() + 1),
      a.batch_sizes());
}
} // namespace neml2
