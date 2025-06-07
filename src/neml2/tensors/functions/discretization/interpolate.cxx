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

#include "neml2/tensors/functions/discretization/interpolate.h"
#include "neml2/tensors/functions/sum.h"
#include "neml2/misc/assertions.h"

namespace neml2::discretization
{
Tensor
interpolate(const Tensor & elem_dofs, const Tensor & basis)
{
  neml_assert(elem_dofs.batch_dim() >= 2,
              "fem_interpolate: batch dimension of elem_dofs must at least 2, got ",
              elem_dofs.batch_dim());
  neml_assert(basis.batch_dim() >= 3,
              "interpolate: base dimension of basis must be at least 3, got ",
              basis.batch_dim());
  neml_assert(elem_dofs.batch_size(-2) == basis.batch_size(-3),
              "interpolate: elem_dofs implies Nelem = ",
              elem_dofs.batch_size(-2),
              ", but basis implies Nelem = ",
              basis.batch_size(-3));
  neml_assert(elem_dofs.batch_size(-1) == basis.batch_size(-2),
              "interpolate: elem_dofs implies Ndofe = ",
              elem_dofs.batch_size(-1),
              ", but basis implies Ndofe = ",
              basis.batch_size(-2));

  return batch_sum(elem_dofs.batch_unsqueeze(-1).base_unsqueeze_to(basis.base_dim()) * basis, -2);
}
} // namespace neml2::discretization
