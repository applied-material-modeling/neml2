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

#include "neml2/tensors/functions/intrsc_intmd_dim_utils.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/sum_to_size.h"
#include "neml2/tensors/Derivative.h"
#include "neml2/tensors/Tensor.h"
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
  auto t = deriv.tensor();

  const auto deriv_ex_is = deriv.extrsc_intmd_sizes();
  const auto var_is = deriv.var_intrsc_intmd_sizes();
  const auto arg_is = deriv.arg_intrsc_intmd_sizes(0);
  const auto var_bs = deriv.var_base_sizes();
  const auto arg_bs = deriv.arg_base_sizes(0);

  if (deriv.is_intrsc_intmd_broadcast())
  {
    neml_assert_dbg(
        at::is_expandable_to(deriv.arg_intrsc_intmd_sizes(0), deriv.var_intrsc_intmd_sizes()),
        "The intrinsic intermediate shape (",
        deriv.arg_intrsc_intmd_sizes(0),
        ") of the argument for derivative '",
        deriv.name(),
        "' is not broadcastable to the variable's intrinsic intermediate shape (",
        deriv.var_intrsc_intmd_sizes(),
        ").");

    // flatten the intrinsic intermediate dimensions
    t = t.intmd_flatten(t.intmd_dim() - deriv.intrsc_intmd_dim());

    // diagonal expand to arguments' intrinsic intermediate dimensions
    t = intmd_diagonalize(t, -1);

    // unflatten to inflated intrinsic intermediate dimensions (variable's intrinsic intermediate
    // dimensions repeated N times)
    const auto inflated_intmd_sizes = utils::add_shapes(deriv_ex_is, var_is, var_is);
    t = t.intmd_reshape(inflated_intmd_sizes);

    // reduce to arguments' intrinsic intermediate dimensions
    const auto padded_intmd_sizes =
        utils::add_shapes(deriv_ex_is, var_is, utils::pad_prepend(arg_is, var_is.size()));
    t = intmd_sum_to_size(t, padded_intmd_sizes)
            .intmd_reshape(utils::add_shapes(deriv_ex_is, var_is, arg_is));
  }

  return pop_intrsc_intmd_dim<2>(t, {var_is.size(), arg_is.size()}, {var_bs, arg_bs}, deriv.name());
}

template <std::size_t N>
Tensor
pop_intrsc_intmd_dim(const Tensor & from,
                     const std::array<std::size_t, N> & intrsc_intmd_dims,
                     const std::array<TensorShapeRef, N> & base_shapes,
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
                                        const std::array<TensorShapeRef, 1> &,
                                        const std::string &);
template Tensor pop_intrsc_intmd_dim<2>(const Tensor &,
                                        const std::array<std::size_t, 2> &,
                                        const std::array<TensorShapeRef, 2> &,
                                        const std::string &);
template Tensor pop_intrsc_intmd_dim<3>(const Tensor &,
                                        const std::array<std::size_t, 3> &,
                                        const std::array<TensorShapeRef, 3> &,
                                        const std::string &);

template <std::size_t N>
Tensor
push_intrsc_intmd_dim(const Tensor & from,
                      const std::array<std::size_t, N> & intrsc_intmd_dims,
                      const std::array<TensorShapeRef, N> & base_shapes,
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
                                         const std::array<TensorShapeRef, 1> &,
                                         const std::string &);
template Tensor push_intrsc_intmd_dim<2>(const Tensor &,
                                         const std::array<std::size_t, 2> &,
                                         const std::array<TensorShapeRef, 2> &,
                                         const std::string &);
template Tensor push_intrsc_intmd_dim<3>(const Tensor &,
                                         const std::array<std::size_t, 3> &,
                                         const std::array<TensorShapeRef, 3> &,
                                         const std::string &);
}
