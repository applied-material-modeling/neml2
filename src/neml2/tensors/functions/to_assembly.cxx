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

#include "neml2/tensors/functions/to_assembly.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
template <std::size_t N>
Tensor
to_assembly(const Tensor & from,
            const std::array<TensorShapeRef, N> & intmd_shapes,
            const std::array<TensorShapeRef, N> & base_shapes,
            [[maybe_unused]] const std::string & debug_name)
{
#ifndef NDEBUG
  // The given tensor should have shape
  //   (dynamic; intmd1, intmd2, ..., intmdN; base1, base2, ..., baseN)
  auto expected_intmd_sizes =
      std::apply([](const auto &... xs) { return utils::add_shapes(xs...); }, intmd_shapes);
  neml_assert_dbg(from.intmd_sizes() == expected_intmd_sizes,
                  "Incompatible intermediate shape for tensor '",
                  debug_name,
                  "', expected intermediate shape is ",
                  expected_intmd_sizes,
                  ", but got ",
                  from.intmd_sizes(),
                  ".");
  auto expected_base_sizes =
      std::apply([](const auto &... xs) { return utils::add_shapes(xs...); }, base_shapes);
  neml_assert_dbg(from.base_sizes() == expected_base_sizes,
                  "Incompatible base shape for tensor '",
                  debug_name,
                  "', expected base shape is ",
                  expected_base_sizes,
                  ", but got ",
                  from.base_sizes());
#endif

  // Generate permutation to move each intmd before each corresponding base
  //
  // For example, for N == 2, the tensor shape is in the form of
  //   (dynamic; intmd1, intmd2; base1, base2)
  //
  // We first move intmd1 before base1:
  //   (dynamic; intmd2; intmd1, base1, base2)
  //
  // Then we move intmd2 before base2:
  //   (dynamic; intmd1, base1, intmd2, base2)
  TensorShape indices(from.dim());
  TensorShape assembly_sizes(N);
  std::iota(indices.begin(), indices.end(), 0);
  auto permutation = indices;
  auto first = permutation.begin() + from.dynamic_dim();
  auto last = first + from.intmd_dim();
  for (std::size_t i = 0; i < N; ++i)
  {
    assembly_sizes[i] = utils::numel(intmd_shapes[i]) * utils::numel(base_shapes[i]);
    if (!intmd_shapes[i].empty())
    {
      auto middle = first + intmd_shapes[i].size();
      std::rotate(first, middle, last);
    }
    last += base_shapes[i].size();
  }

  // Perform the permutation
  auto permuted = Tensor(at::permute(from, permutation), from.dynamic_sizes(), 0);
  return permuted.base_reshape(assembly_sizes);
}

// Explicit instantiations
template Tensor to_assembly<1>(const Tensor &,
                               const std::array<TensorShapeRef, 1> &,
                               const std::array<TensorShapeRef, 1> &,
                               const std::string &);
template Tensor to_assembly<2>(const Tensor &,
                               const std::array<TensorShapeRef, 2> &,
                               const std::array<TensorShapeRef, 2> &,
                               const std::string &);
template Tensor to_assembly<3>(const Tensor &,
                               const std::array<TensorShapeRef, 3> &,
                               const std::array<TensorShapeRef, 3> &,
                               const std::string &);
}
