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

#include "neml2/tensors/functions/vol.h"
#include "neml2/tensors/functions/tr.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/WR2.h"

namespace neml2
{
// Should we ask a check for 2nd order tensors only here?
Tensor
vol(const Tensor & A)
{
  return tr(A) / 3.0 * Tensor::identity(A.base_size(0), A.options());
}

R2
vol(const R2 & A)
{
  return tr(A) / 3.0 * R2::identity(A.options());
}

SR2
vol(const SR2 & A)
{
  return tr(A) / 3.0 * SR2::identity(A.options());
}

WR2
vol(const WR2 & A)
{
  return WR2::zeros(A.options()).batch_expand_as(A);
}
} // namespace neml2
