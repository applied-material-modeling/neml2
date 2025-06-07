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

#include <ATen/ops/zeros.h>

#include "neml2/tensors/functions/discretization/assemble.h"
#include "neml2/misc/assertions.h"

namespace neml2::discretization
{
ATensor
assemble(const ATensor & v_scattered, const ATensor & dof_map, Size ndof)
{
  auto r = at::zeros(ndof, v_scattered.options());
  assemble_(r, v_scattered, dof_map);
  return r;
}

void
assemble_(ATensor & v, const ATensor & v_scattered, const ATensor & dof_map)
{
  neml_assert(v.dim() == 1, "assemble_: v must be a vector, got ", v.sizes());
  neml_assert(v_scattered.sizes() == dof_map.sizes(),
              "assemble_: v_scattered and dof_map must have the same sizes, got ",
              v_scattered.sizes(),
              " and ",
              dof_map.sizes());
  v.scatter_add_(0, dof_map.flatten(), v_scattered.flatten());
}
} // namespace neml2::discretization
