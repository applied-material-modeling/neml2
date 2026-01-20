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

#include "neml2/equation_systems/assembly.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/sum_to_size.h"
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/misc/assertions.h"
#include <ATen/ExpandUtils.h>

namespace neml2
{
template <std::size_t N>
static Tensor
diagonalize_if_needed(const Tensor & t, std::array<TensorShapeRef, N> intmd_shapes)
{
  const auto expected_intmd_sizes =
      std::apply([](const auto &... xs) { return utils::add_shapes(xs...); }, intmd_shapes);
  if (t.intmd_sizes() == expected_intmd_sizes)
    return t;

  auto expanded = t;
  expanded = intmd_diagonalize(expanded.intmd_expand(intmd_shapes[0]).intmd_flatten());
  auto expanded_intmd_sizes = intmd_shapes[0];
  auto padded_intmd_sizes = intmd_shapes[0];
  for (std::size_t i = 1; i < N; ++i)
  {
    expanded_intmd_sizes = utils::add_shapes(expanded_intmd_sizes, intmd_shapes[0]);
    padded_intmd_sizes = utils::add_shapes(
        padded_intmd_sizes, utils::pad_prepend(intmd_shapes[i], intmd_shapes[0].size()));
  }
  return intmd_sum_to_size(expanded.intmd_reshape(expanded_intmd_sizes), padded_intmd_sizes)
      .intmd_reshape(expected_intmd_sizes);
}

template <std::size_t N>
Tensor
to_assembly(const Tensor & from,
            const std::array<TensorShapeRef, N> & intmd_shapes,
            const std::array<TensorShapeRef, N> & base_shapes)
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
  auto expanded = diagonalize_if_needed(from, intmd_shapes);

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
                               const std::array<TensorShapeRef, 1> &,
                               const std::array<TensorShapeRef, 1> &);
template Tensor to_assembly<2>(const Tensor &,
                               const std::array<TensorShapeRef, 2> &,
                               const std::array<TensorShapeRef, 2> &);

template <std::size_t N>
Tensor
from_assembly(const Tensor & from,
              const std::array<TensorShapeRef, N> & intmd_shapes,
              const std::array<TensorShapeRef, N> & base_shapes)
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
                                 const std::array<TensorShapeRef, 1> &,
                                 const std::array<TensorShapeRef, 1> &);
template Tensor from_assembly<2>(const Tensor &,
                                 const std::array<TensorShapeRef, 2> &,
                                 const std::array<TensorShapeRef, 2> &);

static std::vector<Size>
split_sizes(const std::optional<std::vector<TensorShape>> & intmd_shapes,
            const std::vector<TensorShape> & base_shapes)
{
  std::vector<Size> res(base_shapes.size());
  for (std::size_t i = 0; i < base_shapes.size(); ++i)
  {
    res[i] = utils::numel(base_shapes[i]);
    if (intmd_shapes)
      res[i] *= utils::numel((*intmd_shapes)[i]);
  }
  return res;
}

SparseTensorList
disassemble(const Tensor & t,
            const std::optional<std::vector<TensorShape>> & intmd_shapes,
            const std::vector<TensorShape> & base_shapes)
{
  neml_assert_dbg(t.base_dim() == 1, "disassemble expects base dimension of 1, got ", t.base_dim());
  if (intmd_shapes)
    neml_assert_dbg(t.intmd_dim() == 0,
                    "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                    t.intmd_dim());

  const auto ss = split_sizes(intmd_shapes, base_shapes);
  const auto & D = t.dynamic_sizes();
  const auto I = t.intmd_dim();
  const auto vs = t.split(ss, -1);
  const auto n = vs.size();

  SparseTensorList res;
  res.resize(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    auto ti = Tensor(vs[i], D, I);
    // if expected intermediate shapes are not given, we won't touch intermediate dims
    if (!intmd_shapes)
      res[i] = ti.base_reshape(base_shapes[i]);
    else
      res[i] = from_assembly<1>(ti, {(*intmd_shapes)[i]}, {base_shapes[i]});
  }

  return res;
}

Tensor
assemble(const SparseTensorList & t,
         const std::optional<std::vector<TensorShape>> & intmd_shapes,
         const std::vector<TensorShape> & base_shapes)
{
  const auto ss = split_sizes(intmd_shapes, base_shapes);
  const auto n = t.size();

  // If a variable is not found, tensor at that position remains undefined, and all undefined tensor
  // will later be filled with zeros.
  std::vector<Tensor> tf(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    const auto & ti = t[i];
    if (!ti.defined())
      continue;
    const auto tif = intmd_shapes ? to_assembly<1>(ti, {(*intmd_shapes)[i]}, {base_shapes[i]})
                                  : ti.base_flatten();
    tf[i] = tif;
  }

  // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with zeros.
  const auto dynamic_sizes = utils::broadcast_dynamic_sizes(tf);
  const auto intmd_sizes = utils::broadcast_intmd_sizes(tf);
  for (std::size_t i = 0; i < tf.size(); ++i)
  {
    auto & tfi = tf[i];
    if (tfi.defined())
      tfi = tfi.batch_expand(dynamic_sizes, intmd_sizes);
    else
      tfi = Tensor::zeros(dynamic_sizes, intmd_sizes, ss[i], t.options());
  }

  return base_cat(tf, -1);
}

SparseTensorList
disassemble(const Tensor & t,
            const std::optional<std::vector<TensorShape>> & row_intmd_shapes,
            const std::optional<std::vector<TensorShape>> & col_intmd_shapes,
            const std::vector<TensorShape> & row_base_shapes,
            const std::vector<TensorShape> & col_base_shapes)
{
  neml_assert_dbg(t.base_dim() == 2, "disassemble expects base dimension of 2, got ", t.base_dim());
  neml_assert_dbg(
      bool(row_intmd_shapes) == bool(col_intmd_shapes),
      "Either both or neither of row_intmd_shapes and col_intmd_shapes should be given.");
  if (row_intmd_shapes || col_intmd_shapes)
    neml_assert_dbg(t.intmd_dim() == 0,
                    "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                    t.intmd_dim());

  const auto row_ss = split_sizes(row_intmd_shapes, row_base_shapes);
  const auto col_ss = split_sizes(col_intmd_shapes, col_base_shapes);
  const auto & D = t.dynamic_sizes();
  const auto I = t.intmd_dim();
  const auto m = row_ss.size();
  const auto n = col_ss.size();

  SparseTensorList res;
  res.resize(m * n);
  // split into rows
  auto t_rows = t.split(row_ss, -2);
  for (std::size_t i = 0; i < m; ++i)
  {
    // split into columns
    auto t_cols = t_rows[i].split(col_ss, -1);
    for (std::size_t j = 0; j < n; ++j)
    {
      const auto & tij = Tensor(t_cols[j], D, I);
      // if expected intermediate shapes are not given, we won't touch intermediate dims
      if (!row_intmd_shapes || !col_intmd_shapes)
        res[i * n + j] =
            tij.base_reshape(utils::add_shapes(row_base_shapes[i], col_base_shapes[j]));
      else
        res[i * n + j] = from_assembly<2>(tij,
                                          {(*row_intmd_shapes)[i], (*col_intmd_shapes)[j]},
                                          {row_base_shapes[i], col_base_shapes[j]});
    }
  }

  return res;
}

Tensor
assemble(const SparseTensorList & t,
         const std::optional<std::vector<TensorShape>> & row_intmd_shapes,
         const std::optional<std::vector<TensorShape>> & col_intmd_shapes,
         const std::vector<TensorShape> & row_base_shapes,
         const std::vector<TensorShape> & col_base_shapes)
{
  neml_assert_dbg(
      bool(row_intmd_shapes) == bool(col_intmd_shapes),
      "Either both or neither of row_intmd_shapes and col_intmd_shapes should be given.");

  const bool intmd = row_intmd_shapes && col_intmd_shapes;
  const auto row_ss = split_sizes(row_intmd_shapes, row_base_shapes);
  const auto col_ss = split_sizes(col_intmd_shapes, col_base_shapes);
  const auto m = row_ss.size();
  const auto n = col_ss.size();

  // column ndof
  auto col_ndof = std::accumulate(col_ss.begin(), col_ss.end(), Size(0), std::plus<>());

  // If a variable is not found, tensor at that position remains undefined, and all undefined tensor
  // will later be filled with zeros.
  std::vector<Tensor> tf_rows(m);
  for (std::size_t i = 0; i < m; ++i)
  {
    std::vector<Tensor> tf_cols(n);
    for (std::size_t j = 0; j < n; ++j)
    {
      const auto & tij = t[i * n + j];
      if (!tij.defined())
        continue;
      const auto tijf = intmd ? to_assembly<2>(tij,
                                               {(*row_intmd_shapes)[i], (*col_intmd_shapes)[j]},
                                               {row_base_shapes[i], col_base_shapes[j]})
                              : tij.base_reshape({utils::numel(row_base_shapes[i]),
                                                  utils::numel(col_base_shapes[j])});
      tf_cols[j] = tijf;
    }

    // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with
    // zeros.
    const auto dynamic_sizes = utils::broadcast_dynamic_sizes(tf_cols);
    const auto intmd_sizes = utils::broadcast_intmd_sizes(tf_cols);
    for (std::size_t j = 0; j < n; ++j)
    {
      auto & tfij = tf_cols[j];
      if (tfij.defined())
        tfij = tfij.batch_expand(dynamic_sizes, intmd_sizes);
      else
        tfij = Tensor::zeros(dynamic_sizes, intmd_sizes, {row_ss[i], col_ss[j]}, t.options());
    }

    tf_rows[i] = base_cat(tf_cols, -1);
  }

  // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with zeros.
  const auto dynamic_sizes = utils::broadcast_dynamic_sizes(tf_rows);
  const auto intmd_sizes = utils::broadcast_intmd_sizes(tf_rows);
  for (std::size_t i = 0; i < m; ++i)
  {
    auto & tfi = tf_rows[i];
    if (tfi.defined())
      tfi = tfi.batch_expand(dynamic_sizes, intmd_sizes);
    else
      tfi = Tensor::zeros(dynamic_sizes, intmd_sizes, {row_ss[i], col_ndof}, t.options());
  }

  return base_cat(tf_rows, -2);
}
} // namespace neml2
