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

#include "neml2/tensors/functions/fem.h"
#include "neml2/tensors/functions/sum.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
Tensor
fem_scatter(const Tensor & v, const Tensor & dof_map)
{
  neml_assert_dbg(!v.batched(), "fem_scatter: v must be unbatched.");
  neml_assert_dbg(dof_map.batch_dim() == 1,
                  "fem_scatter: batch dimension of dof_map must be 1, got ",
                  dof_map.batch_dim());
  neml_assert_dbg(dof_map.base_dim() == 2,
                  "fem_scatter: base dimension of dof_map must be 2, got ",
                  dof_map.base_dim());

  auto vals = v.index({at::Tensor(dof_map)});
  return Tensor(vals, dof_map.batch_sizes());
}

Tensor
fem_interpolate(const Tensor & elem_dofs, const Tensor & basis)
{
  neml_assert_dbg(elem_dofs.batch_dim() == 1,
                  "fem_interpolate: batch dimension of elem_dofs must be 1, got ",
                  elem_dofs.batch_dim());
  neml_assert_dbg(elem_dofs.base_dim() == 2,
                  "fem_interpolate: base dimension of elem_dofs must be 2, got ",
                  elem_dofs.base_dim());
  neml_assert_dbg(
      basis.base_dim() >= 1,
      "fem_interpolate: base dimension of basis must be greater than or equal to 1, got ",
      basis.base_dim());
  neml_assert_dbg(elem_dofs.base_size(0) == basis.base_size(0),
                  "fem_interpolate: sizes of the first base dimension of elem_dofs and basis must "
                  "be equal, got ",
                  elem_dofs.base_size(0),
                  " and ",
                  basis.base_size(0));

  return base_sum(elem_dofs.base_unsqueeze_to(basis.base_dim() + 1) * basis.base_unsqueeze(1), 0);
}

Tensor
fem_assemble(const ATensor & v_scattered, const ATensor & dof_map, Size ndof)
{
  neml_assert_dbg(v_scattered.sizes() == dof_map.sizes(),
                  "fem_assemble: v_scattered and dof_map must have the same sizes, got ",
                  v_scattered.sizes(),
                  " and ",
                  dof_map.sizes());
  auto r = Tensor::zeros(ndof, v_scattered.options());
  r.scatter_add_(0, dof_map.flatten(), v_scattered.flatten());
  return Tensor(r, 0);
}
} // namespace neml2
