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

#include "neml2/tensors/functions/chain_rule.h"
#include "neml2/tensors/Derivative.h"
#include "neml2/tensors/functions/mm.h"
#include "neml2/tensors/functions/einsum.h"
#include "neml2/tensors/functions/sum.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2
{

Derivative<1>
chain_rule(const Derivative<1> & dy_du, const Derivative<1> & du_dx)
{
  // Intrinsic intermediate dimensions
  const auto y_intrsc_intmd_dim = dy_du.var_intrsc_intmd_dim();
  const auto u_intrsc_intmd_dim = dy_du.arg_intrsc_intmd_dim(0);
  const auto u2_intrsc_intmd_dim = du_dx.var_intrsc_intmd_dim();
  const auto x_intrsc_intmd_dim = du_dx.arg_intrsc_intmd_dim(0);

  // Reinterpret derivatives if intrinsic intermediate dimensions do not match
  if (u_intrsc_intmd_dim > u2_intrsc_intmd_dim)
    return chain_rule(dy_du, du_dx.reinterpret(u_intrsc_intmd_dim - u2_intrsc_intmd_dim));
  else if (u_intrsc_intmd_dim < u2_intrsc_intmd_dim)
    return chain_rule(dy_du.reinterpret(u2_intrsc_intmd_dim - u_intrsc_intmd_dim), du_dx);

  // Base sizes
  const auto y_base_sizes = dy_du.var_base_sizes();
  const auto u_base_sizes = dy_du.arg_base_sizes(0);
  const auto x_base_sizes = du_dx.arg_base_sizes(0);
  neml_assert_dbg(u_base_sizes == du_dx.var_base_sizes(), "Incompatible base sizes for chain rule");

  // Reshape the derivatives to 2D matrices
  const auto ny = utils::numel(y_base_sizes);
  const auto nu = utils::numel(u_base_sizes);
  const auto nx = utils::numel(x_base_sizes);
  auto dy_du_f = dy_du.tensor().base_reshape({ny, nu});
  auto du_dx_f = du_dx.tensor().base_reshape({nu, nx});

  // Align dependent intermediate dimensions (for matrix multiplication)
  if (!dy_du.is_intrsc_intmd_broadcast())
    du_dx_f = du_dx_f.intmd_unsqueeze(-du_dx.intrsc_intmd_dim() - 1, Size(y_intrsc_intmd_dim));
  if (!du_dx.is_intrsc_intmd_broadcast())
    dy_du_f = dy_du_f.intmd_unsqueeze(-1, Size(x_intrsc_intmd_dim));

  // Apply chain rule via matrix multiplication
  auto dy_dx_f = neml2::mm(dy_du_f, du_dx_f);

  // Reduce intrinsic intermediate dimensions for u
  if (u_intrsc_intmd_dim > 0 && !dy_du.is_intrsc_intmd_broadcast() &&
      !du_dx.is_intrsc_intmd_broadcast())
  {
    TensorShape reduce_dims(u_intrsc_intmd_dim);
    std::iota(reduce_dims.begin(), reduce_dims.end(), -u_intrsc_intmd_dim - x_intrsc_intmd_dim);
    dy_dx_f = neml2::intmd_sum(dy_dx_f, reduce_dims, false);
  }

  // Reshape back to original base sizes
  auto dy_dx = Derivative<1>(y_intrsc_intmd_dim + x_intrsc_intmd_dim,
                             {std::size_t(y_intrsc_intmd_dim), std::size_t(x_intrsc_intmd_dim)},
                             {dy_du.var_intmd_sizes(), du_dx.arg_intmd_sizes(0)},
                             {y_base_sizes, x_base_sizes},
                             dy_du.var_name(),
                             {du_dx.arg_name(0)});
  dy_dx = dy_dx_f.base_reshape(utils::add_shapes(y_base_sizes, x_base_sizes));

  return dy_dx;
}

Derivative<2>
chain_rule(const Derivative<2> & d2y_du1u2,
           const Derivative<1> * du1_dx1,
           const Derivative<1> * du2_dx2)
{
  // Base sizes
  const auto y_base_sizes = d2y_du1u2.var_base_sizes();
  const auto u1_base_sizes = d2y_du1u2.arg_base_sizes(0);
  const auto u2_base_sizes = d2y_du1u2.arg_base_sizes(1);
  const auto x1_base_sizes = du1_dx1 ? du1_dx1->arg_base_sizes(0) : TensorShapeRef{};
  const auto x2_base_sizes = du2_dx2 ? du2_dx2->arg_base_sizes(0) : TensorShapeRef{};
  if (du1_dx1)
    neml_assert_dbg(u1_base_sizes == du1_dx1->var_base_sizes(),
                    "Incompatible base sizes for chain rule between '",
                    d2y_du1u2.name(),
                    "' and '",
                    du1_dx1->name(),
                    "'.");
  if (du2_dx2)
    neml_assert_dbg(u2_base_sizes == du2_dx2->var_base_sizes(),
                    "Incompatible base sizes for chain rule between '",
                    d2y_du1u2.name(),
                    "' and '",
                    du2_dx2->name(),
                    "'.");

  // Reshape the derivatives to 2D/3D matrices
  const auto ny = utils::numel(y_base_sizes);
  const auto nu1 = utils::numel(u1_base_sizes);
  const auto nu2 = utils::numel(u2_base_sizes);
  const auto nx1 = utils::numel(x1_base_sizes);
  const auto nx2 = utils::numel(x2_base_sizes);
  auto d2y_du1u2_f = d2y_du1u2.tensor().base_reshape({ny, nu1, nu2});
  Tensor du1_dx1_f, du2_dx2_f;
  if (du1_dx1)
    du1_dx1_f = du1_dx1->tensor().base_reshape({nu1, nx1});
  if (du2_dx2)
    du2_dx2_f = du2_dx2->tensor().base_reshape({nu2, nx2});

  // Apply chain rule via matrix multiplication
  Tensor d2y_dx1x2_f;
  if (du1_dx1 && du2_dx2)
    d2y_dx1x2_f = neml2::einsum("...ipq,...pj,...qk", {d2y_du1u2_f, du1_dx1_f, du2_dx2_f});
  else if (du1_dx1)
    d2y_dx1x2_f = neml2::einsum("...ipk,...pj", {d2y_du1u2_f, du1_dx1_f});
  else if (du2_dx2)
    d2y_dx1x2_f = neml2::einsum("...ijq,...qk", {d2y_du1u2_f, du2_dx2_f});

  // Reshape back to original base sizes
  auto d2y_dx1x2 =
      Derivative<2>(0,
                    {0, 0, 0},
                    {d2y_du1u2.var_intmd_sizes(),
                     du1_dx1 ? du1_dx1->arg_intmd_sizes(0) : d2y_du1u2.arg_intmd_sizes(1),
                     du2_dx2 ? du2_dx2->arg_intmd_sizes(0) : d2y_du1u2.arg_intmd_sizes(1)},
                    {y_base_sizes, x1_base_sizes, x2_base_sizes});
  d2y_dx1x2 =
      d2y_dx1x2_f.base_reshape(utils::add_shapes(y_base_sizes, x1_base_sizes, x2_base_sizes));
  return d2y_dx1x2;
}

Derivative<2>
chain_rule(const Derivative<1> & dy_du, const Derivative<2> & d2u_dx1x2)
{
  // Base sizes
  const auto y_base_sizes = dy_du.var_base_sizes();
  const auto u_base_sizes = dy_du.arg_base_sizes(0);
  const auto x1_base_sizes = d2u_dx1x2.arg_base_sizes(0);
  const auto x2_base_sizes = d2u_dx1x2.arg_base_sizes(1);
  neml_assert_dbg(u_base_sizes == d2u_dx1x2.var_base_sizes(),
                  "Incompatible base sizes for chain rule between '",
                  dy_du.name(),
                  "' and '",
                  d2u_dx1x2.name(),
                  "'.");

  // Reshape the derivatives to 2D/3D matrices
  const auto ny = utils::numel(y_base_sizes);
  const auto nu = utils::numel(u_base_sizes);
  const auto nx1 = utils::numel(x1_base_sizes);
  const auto nx2 = utils::numel(x2_base_sizes);
  auto dy_du_f = dy_du.tensor().base_reshape({ny, nu});
  auto d2u_dx1x2_f = d2u_dx1x2.tensor().base_reshape({nu, nx1, nx2});

  // Apply chain rule via matrix multiplication
  auto d2y_dx1x2_f = neml2::einsum("...ip,...pjk", {dy_du_f, d2u_dx1x2_f});

  // Reshape back to original base sizes
  auto d2y_dx1x2 = Derivative<2>(
      0,
      {0, 0, 0},
      {dy_du.var_intmd_sizes(), d2u_dx1x2.arg_intmd_sizes(0), d2u_dx1x2.arg_intmd_sizes(1)},
      {y_base_sizes, x1_base_sizes, x2_base_sizes});
  d2y_dx1x2 =
      d2y_dx1x2_f.base_reshape(utils::add_shapes(y_base_sizes, x1_base_sizes, x2_base_sizes));
  return d2y_dx1x2;
}

} // namespace neml2
