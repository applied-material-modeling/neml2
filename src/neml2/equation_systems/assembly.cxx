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

#include <ATen/ExpandUtils.h>
#include <cstddef>

#include "neml2/equation_systems/assembly.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Derivative.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2
{

Tensor
pop_intrsc_intmd_dim(const Tensor & t, Size dim)
{
  if (dim == 0)
    return t;
  neml_assert_dbg(dim <= t.intmd_dim(),
                  "Unable to pop ",
                  dim,
                  " intermediate dimensions from a tensor with intermediate dimension ",
                  t.intmd_dim(),
                  ".");
  return Tensor(t, t.dynamic_sizes(), t.intmd_dim() - dim);
}

Tensor
push_intrsc_intmd_dim(const Tensor & t, Size dim)
{
  if (dim == 0)
    return t;
  neml_assert_dbg(dim <= t.base_dim(),
                  "Unable to push ",
                  dim,
                  " intermediate dimensions from a tensor with base dimension ",
                  t.base_dim(),
                  ".");
  return Tensor(t, t.dynamic_sizes(), t.intmd_dim() + dim);
}

Tensor
pop_intrsc_intmd_dim(const Derivative<1> & deriv)
{
  auto t = deriv.fullify();

  const auto var_id = std::size_t(deriv.var_intrsc_intmd_dim());
  const auto arg_id = std::size_t(deriv.arg_intrsc_intmd_dim(0));
  const auto var_bs = TensorShape(deriv.var_base_sizes());
  const auto arg_bs = TensorShape(deriv.arg_base_sizes(0));

  return pop_intrsc_intmd_dim<2>(t, {var_id, arg_id}, {var_bs, arg_bs}, deriv.name());
}

template <std::size_t N>
Tensor
pop_intrsc_intmd_dim(const Tensor & from,
                     const std::array<std::size_t, N> & intrsc_intmd_dims,
                     const std::array<TensorShape, N> & base_shapes,
                     [[maybe_unused]] const std::string & debug_name)
{
  const auto total_id = utils::sum_array(intrsc_intmd_dims);

#ifndef NDEBUG
  // The given tensor should have shape
  //   (*dynamic; *indep_intmd, *dep_intmd; *base)
  //
  // where dep_intmd = (*dep_intmd_0, ..., *dep_intmd_{N-1})
  //            base = (*base_0, ..., *base_{N-1})
  neml_assert_dbg(from.intmd_dim() >= Size(total_id),
                  "Number of intermediate dimensions (",
                  from.intmd_dim(),
                  ") is less than the total number of intermediate dimensions to pop (",
                  total_id,
                  ") for tensor '",
                  debug_name,
                  "'.");
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

  // Move each intrinsic dimension to its corresponding position in the base
  auto src_dim = -total_id;
  if (src_dim != 0)
    src_dim = utils::normalize_dim(src_dim, from.dynamic_dim(), from.batch_dim());
  auto dest_dim = from.batch_dim() - 1;
  auto raw = ATensor(from);
  for (std::size_t i = 0; i < N; i++)
  {
    for (std::size_t j = 0; j < intrsc_intmd_dims[i]; j++)
      raw = raw.movedim(src_dim, dest_dim);
    dest_dim += base_shapes[i].size();
  }

  return Tensor(raw, from.dynamic_sizes(), from.intmd_dim() - total_id);
}

// Explicit instantiations
template Tensor pop_intrsc_intmd_dim<1>(const Tensor &,
                                        const std::array<std::size_t, 1> &,
                                        const std::array<TensorShape, 1> &,
                                        const std::string &);
template Tensor pop_intrsc_intmd_dim<2>(const Tensor &,
                                        const std::array<std::size_t, 2> &,
                                        const std::array<TensorShape, 2> &,
                                        const std::string &);
template Tensor pop_intrsc_intmd_dim<3>(const Tensor &,
                                        const std::array<std::size_t, 3> &,
                                        const std::array<TensorShape, 3> &,
                                        const std::string &);

template <std::size_t N>
Tensor
push_intrsc_intmd_dim(const Tensor & from,
                      const std::array<std::size_t, N> & intrsc_intmd_dims,
                      const std::array<TensorShape, N> & base_shapes,
                      [[maybe_unused]] const std::string & debug_name)
{
  // The given tensor should have shape
  //   (*dynamic; *indep_intmd; *dep_intmd_0, *base_0, ..., *dep_intmd_{N-1}, *base_{N-1})
  // We need to move the dependent intermediate dimension from base to intmd
  auto src_dim = from.batch_dim();
  auto dest_dim = from.batch_dim();
  auto raw = ATensor(from);
  Size total_id = 0;
  for (std::size_t i = 0; i < N; i++)
  {
    for (std::size_t j = 0; j < intrsc_intmd_dims[i]; j++)
      raw = raw.movedim(src_dim++, dest_dim++);
    src_dim += base_shapes[i].size();
    total_id += intrsc_intmd_dims[i];
  }
  return Tensor(raw, from.dynamic_sizes(), from.intmd_dim() + total_id);
}

// Explicit instantiations
template Tensor push_intrsc_intmd_dim<1>(const Tensor &,
                                         const std::array<std::size_t, 1> &,
                                         const std::array<TensorShape, 1> &,
                                         const std::string &);
template Tensor push_intrsc_intmd_dim<2>(const Tensor &,
                                         const std::array<std::size_t, 2> &,
                                         const std::array<TensorShape, 2> &,
                                         const std::string &);
template Tensor push_intrsc_intmd_dim<3>(const Tensor &,
                                         const std::array<std::size_t, 3> &,
                                         const std::array<TensorShape, 3> &,
                                         const std::string &);

template <std::size_t N>
Tensor
to_assembly(const Tensor & from,
            const std::array<TensorShape, N> & intmd_shapes,
            const std::array<TensorShape, N> & base_shapes)
{
#ifndef NDEBUG
  // The given tensor should have shape
  //   (dynamic; intmd1, intmd2, ..., intmdN; base1, base2, ..., baseN)
  const auto expected_intmd_sizes =
      std::apply([](const auto &... xs) { return utils::add_shapes(xs...); }, intmd_shapes);
  neml_assert_dbg(from.intmd_sizes() == expected_intmd_sizes ||
                      at::is_expandable_to(from.intmd_sizes(), intmd_shapes[0]),
                  "Incompatible intermediate shape, expected intermediate shape to be either ",
                  expected_intmd_sizes,
                  ", or expandable to ",
                  intmd_shapes[0],
                  ", got ",
                  from.intmd_sizes(),
                  ".");
  const auto expected_base_sizes =
      std::apply([](const auto &... xs) { return utils::add_shapes(xs...); }, base_shapes);
  neml_assert_dbg(from.base_sizes() == expected_base_sizes,
                  "Incompatible base shape, expected base shape is ",
                  expected_base_sizes,
                  ", got ",
                  from.base_sizes());
#endif

  // Expand intermediate sizes if needed
  auto expanded = fullify<N>(from, from.intmd_dim(), intmd_shapes);

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
  TensorShape indices(expanded.dim());
  TensorShape assembly_sizes(N);
  std::iota(indices.begin(), indices.end(), 0);
  auto permutation = indices;
  auto first = permutation.begin() + expanded.dynamic_dim();
  auto last = first + expanded.intmd_dim();
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
  auto permuted = Tensor(at::permute(expanded, permutation), expanded.dynamic_sizes(), 0);
  return permuted.base_reshape(assembly_sizes);
}

// Explicit instantiations
template Tensor to_assembly<1>(const Tensor &,
                               const std::array<TensorShape, 1> &,
                               const std::array<TensorShape, 1> &);
template Tensor to_assembly<2>(const Tensor &,
                               const std::array<TensorShape, 2> &,
                               const std::array<TensorShape, 2> &);

template <std::size_t N>
Tensor
from_assembly(const Tensor & from,
              const std::array<TensorShape, N> & intmd_shapes,
              const std::array<TensorShape, N> & base_shapes)
{
#ifndef NDEBUG
  TensorShape assembly_sizes(N);
  for (std::size_t i = 0; i < N; i++)
    assembly_sizes[i] = utils::numel(intmd_shapes[i]) * utils::numel(base_shapes[i]);
  neml_assert_dbg(from.intmd_dim() == 0,
                  "Tensor in assembly format should have no intermediate dimensions");
  neml_assert_dbg(assembly_sizes == from.base_sizes(),
                  "Incompatible base shape, expected base shape ",
                  assembly_sizes,
                  ", got ",
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
                                 const std::array<TensorShape, 1> &,
                                 const std::array<TensorShape, 1> &);
template Tensor from_assembly<2>(const Tensor &,
                                 const std::array<TensorShape, 2> &,
                                 const std::array<TensorShape, 2> &);

} // namespace neml2
