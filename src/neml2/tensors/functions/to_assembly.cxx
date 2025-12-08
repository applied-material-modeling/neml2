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
            const std::array<TensorShapeRef, N> & dep_intmd_shapes,
            const std::array<TensorShapeRef, N> & base_shapes,
            [[maybe_unused]] const std::string & debug_name)
{
#ifndef NDEBUG
  // The given tensor should have shape
  //   (*dynamic; *indep_intmd, *dep_intmd; *base)
  //
  // where dep_intmd = (*dep_intmd_0, ..., *dep_intmd_{N-1})
  //            base = (*base_0, ..., *base_{N-1})
  const auto total_dep_intmd_sizes =
      std::apply([](const auto &... xs) { return utils::add_shapes(xs...); }, dep_intmd_shapes);
  const auto indep_intmd_dim = from.intmd_dim() - Size(total_dep_intmd_sizes.size());
  neml_assert_dbg(indep_intmd_dim >= 0,
                  "Incompatible intermediate dimension for tensor '",
                  debug_name,
                  "', expected at least ",
                  total_dep_intmd_sizes.size(),
                  " (dependent intermediate dimensions), got ",
                  from.intmd_dim(),
                  ".");
  neml_assert_dbg(from.intmd_sizes().slice(indep_intmd_dim) == total_dep_intmd_sizes,
                  "Incompatible intermediate shape for tensor '",
                  debug_name,
                  "', expected trailing intermediate shape ",
                  total_dep_intmd_sizes,
                  ", but got ",
                  from.intmd_sizes(),
                  ".");
  const auto total_base_sizes =
      std::apply([](const auto &... xs) { return utils::add_shapes(xs...); }, base_shapes);
  neml_assert_dbg(from.base_sizes() == total_base_sizes,
                  "Incompatible base shape for tensor '",
                  debug_name,
                  "', expected base shape ",
                  total_base_sizes,
                  ", but got ",
                  from.base_sizes());
#endif

  const auto total_dep_intmd_dim =
      std::accumulate(dep_intmd_shapes.begin(),
                      dep_intmd_shapes.end(),
                      std::size_t{0},
                      [](std::size_t a, const auto & x) { return a + x.size(); });

  // Move each dependent dimension to its corresponding position in the base
  auto src_dim = -total_dep_intmd_dim;
  if (src_dim != 0)
    src_dim = utils::normalize_dim(src_dim, from.dynamic_dim(), from.batch_dim());
  auto dest_dim = from.batch_dim() - 1;
  auto raw = ATensor(from);
  auto assembly_sizes = std::array<Size, N>{};
  for (std::size_t i = 0; i < N; i++)
  {
    for (std::size_t j = 0; j < dep_intmd_shapes[i].size(); j++)
      raw = raw.movedim(src_dim, dest_dim);
    dest_dim += base_shapes[i].size();
    assembly_sizes[i] = utils::numel(dep_intmd_shapes[i]) * utils::numel(base_shapes[i]);
  }

  return Tensor(raw, from.dynamic_sizes(), from.intmd_dim() - total_dep_intmd_dim);
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
