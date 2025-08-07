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
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
template <std::size_t N>
Tensor
from_assembly(const Tensor & from,
              const std::array<TensorShapeRef, N> & intmd_shapes,
              const std::array<TensorShapeRef, N> & base_shapes,
              const std::string & debug_name)
{
#ifndef NDEBUG
  TensorShape assembly_sizes(N);
  for (std::size_t i = 0; i < N; i++)
    assembly_sizes[i] = utils::numel(intmd_shapes[i]) * utils::numel(base_shapes[i]);
  neml_assert_dbg(from.intmd_dim() == 0,
                  "Tensor in assembly format should have no intermediate dimensions");
  neml_assert_dbg(assembly_sizes == from.base_sizes(),
                  "Incompatible base shape for tensor '",
                  debug_name,
                  "', expected base shape is ",
                  assembly_sizes,
                  ", but got ",
                  from.base_sizes());
#endif

  // Generate the unflattened base shape
  TensorShape unfl_sizes;
  for (std::size_t i = 0; i < N; i++)
  {
    unfl_sizes.insert(unfl_sizes.end(), intmd_shapes[i].begin(), intmd_shapes[i].end());
    unfl_sizes.insert(unfl_sizes.end(), base_shapes[i].begin(), base_shapes[i].end());
  }

  // Unflatten base
  auto unfl = from.base_reshape(unfl_sizes);

  // Generate permutation for intmd dimensions
  //
  // For example, for N == 3, the tensor shape is in the form of
  //   (dynamic; intmd1, base1, intmd2, base2, intmd3, base3)
  //
  // We first move intmd1 after dynamic:
  //   (dynamic; intmd1; base1, intmd2, base2, intmd3, base3)
  //
  // Then we move intmd2 after intmd1:
  //   (dynamic; intmd1, intmd2; base1, base2, intmd3, base3)
  //
  // Finally, we move intmd3 after intmd2:
  //   (dynamic; intmd1, intmd2, intmd3; base1, base2, base3)
  Size intmd_dim = intmd_shapes[0].size();
  TensorShape indices(unfl.dim());
  std::iota(indices.begin(), indices.end(), 0);
  auto permutation = indices;
  auto first = permutation.begin() + unfl.dynamic_dim() + intmd_shapes[0].size();
  auto middle = first + base_shapes[0].size();
  for (std::size_t i = 1; i < N; ++i)
  {
    if (!intmd_shapes[i].empty())
    {
      intmd_dim += intmd_shapes[i].size();
      auto last = middle + intmd_shapes[i].size();
      std::rotate(first, middle, last);
      first += intmd_shapes[i].size();
      middle += intmd_shapes[i].size();
    }
    middle += base_shapes[i].size();
  }

  // Perform the permutation
  return Tensor(at::permute(unfl, permutation), unfl.dynamic_sizes(), intmd_dim);
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
