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

#include "neml2/tensors/functions/from_assembly.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
template <std::size_t N>
Tensor
from_assembly(const Tensor & from,
              const std::array<TensorShapeRef, N> & dep_intmd_shapes,
              const std::array<TensorShapeRef, N> & base_shapes,
              [[maybe_unused]] const std::string & debug_name)
{
#ifndef NDEBUG
  TensorShape assembly_sizes(N);
  for (std::size_t i = 0; i < N; i++)
    assembly_sizes[i] = utils::numel(dep_intmd_shapes[i]) * utils::numel(base_shapes[i]);
  neml_assert_dbg(assembly_sizes == from.base_sizes(),
                  "Incompatible base shape for tensor '",
                  debug_name,
                  "', expected base shape ",
                  assembly_sizes,
                  ", but got ",
                  from.base_sizes());
#endif

  // Generate the unflattened base shape
  TensorShape unfl_sizes;
  for (std::size_t i = 0; i < N; i++)
  {
    unfl_sizes.insert(unfl_sizes.end(), dep_intmd_shapes[i].begin(), dep_intmd_shapes[i].end());
    unfl_sizes.insert(unfl_sizes.end(), base_shapes[i].begin(), base_shapes[i].end());
  }

  // Unflatten base
  auto unfl = from.base_reshape(unfl_sizes);

  // The given tensor should have shape
  //   (*dynamic; *indep_intmd; *dep_intmd_0, *base_0, ..., *dep_intmd_{N-1}, *base_{N-1})
  // We need to move the dependent intermediate dimension from base to intmd
  auto src_dim = from.batch_dim();
  auto dest_dim = from.batch_dim();
  auto raw = ATensor(unfl);
  Size total_dep_intmd_dim = 0;
  for (std::size_t i = 0; i < N; i++)
  {
    for (std::size_t j = 0; j < dep_intmd_shapes[i].size(); j++)
      raw = raw.movedim(src_dim++, dest_dim++);
    src_dim += base_shapes[i].size();
    total_dep_intmd_dim += dep_intmd_shapes[i].size();
  }
  return Tensor(raw, from.dynamic_sizes(), from.intmd_dim() + total_dep_intmd_dim);
}

// Explicit instantiations
template Tensor from_assembly<1>(const Tensor &,
                                 const std::array<TensorShapeRef, 1> &,
                                 const std::array<TensorShapeRef, 1> &,
                                 const std::string &);
template Tensor from_assembly<2>(const Tensor &,
                                 const std::array<TensorShapeRef, 2> &,
                                 const std::array<TensorShapeRef, 2> &,
                                 const std::string &);
template Tensor from_assembly<3>(const Tensor &,
                                 const std::array<TensorShapeRef, 3> &,
                                 const std::array<TensorShapeRef, 3> &,
                                 const std::string &);
}
