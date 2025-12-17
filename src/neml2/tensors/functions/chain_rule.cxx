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
  const auto y_intrsc_intmd_dim = dy_du.intrsc_intmd_dim(0);
  const auto u_intrsc_intmd_dim = dy_du.intrsc_intmd_dim(1);
  const auto x_intrsc_intmd_dim = du_dx.intrsc_intmd_dim(1);
  neml_assert_dbg(u_intrsc_intmd_dim == du_dx.intrsc_intmd_dim(0),
                  "Incompatible dependent intermediate dimensions for chain rule");

  // Base sizes
  const auto y_base_sizes = dy_du.base_sizes(0);
  const auto u_base_sizes = dy_du.base_sizes(1);
  const auto x_base_sizes = du_dx.base_sizes(1);
  neml_assert_dbg(u_base_sizes == du_dx.base_sizes(0), "Incompatible base sizes for chain rule");

  // Reshape the derivatives to 2D matrices
  auto dy_du_f =
      dy_du.tensor().base_reshape({utils::numel(y_base_sizes), utils::numel(u_base_sizes)});
  auto du_dx_f =
      du_dx.tensor().base_reshape({utils::numel(u_base_sizes), utils::numel(x_base_sizes)});

  // Align dependent intermediate dimensions (for matrix multiplication)
  if (!du_dx.is_intrsc_intmd_broadcast())
    dy_du_f = dy_du_f.intmd_unsqueeze(-1, Size(x_intrsc_intmd_dim));
  du_dx_f = du_dx_f.intmd_unsqueeze(Size(-u_intrsc_intmd_dim - x_intrsc_intmd_dim - 1),
                                    Size(y_intrsc_intmd_dim));

  // Apply chain rule via matrix multiplication
  auto dy_dx_f = neml2::mm(dy_du_f, du_dx_f);

  // Reduce intrinsic intermediate dimensions for u
  if (u_intrsc_intmd_dim > 0 && !du_dx.is_intrsc_intmd_broadcast())
  {
    TensorShape reduce_dims(u_intrsc_intmd_dim);
    std::iota(reduce_dims.begin(), reduce_dims.end(), -u_intrsc_intmd_dim - x_intrsc_intmd_dim);
    dy_dx_f = neml2::intmd_sum(dy_dx_f, reduce_dims, false);
  }

  // Reshape back to original base sizes
  auto dy_dx = Derivative<1>({dy_du.intrsc_intmd_sizes(0), du_dx.intrsc_intmd_sizes(1)},
                             {y_base_sizes, x_base_sizes},
                             "<anonymous chain rule>",
                             y_intrsc_intmd_dim + x_intrsc_intmd_dim);
  dy_dx = dy_dx_f.base_reshape(utils::add_shapes(y_base_sizes, x_base_sizes));
  return dy_dx;
}

Derivative<2>
chain_rule(const Derivative<2> & d2y_du1u2,
           const Derivative<1> * du1_dx1,
           const Derivative<1> * du2_dx2)
{
  // // Dependent intermediate dimensions
  // const auto y_intrsc_intmd_dim = d2y_du1u2.intrsc_intmd_sizes(0).size();
  // const auto u1_intrsc_intmd_dim = d2y_du1u2.intrsc_intmd_sizes(1).size();
  // const auto u2_intrsc_intmd_dim = d2y_du1u2.intrsc_intmd_sizes(2).size();
  // const auto x1_intrsc_intmd_dim = du1_dx1 ? du1_dx1->intrsc_intmd_sizes(1).size() : 0;
  // const auto x2_intrsc_intmd_dim = du2_dx2 ? du2_dx2->intrsc_intmd_sizes(2).size() : 0;
  // if (du1_dx1)
  //   neml_assert_dbg(u1_intrsc_intmd_dim >= du1_dx1->intrsc_intmd_sizes(0).size(),
  //                   "Incompatible dependent intermediate dimensions for chain rule");
  // if (du2_dx2)
  //   neml_assert_dbg(u2_intrsc_intmd_dim >= du2_dx2->intrsc_intmd_sizes(0).size(),
  //                   "Incompatible dependent intermediate dimensions for chain rule");

  // Base sizes
  const auto y_base_sizes = d2y_du1u2.base_sizes(0);
  const auto u1_base_sizes = d2y_du1u2.base_sizes(1);
  const auto u2_base_sizes = d2y_du1u2.base_sizes(2);
  const auto x1_base_sizes = du1_dx1 ? du1_dx1->base_sizes(1) : TensorShapeRef{};
  const auto x2_base_sizes = du2_dx2 ? du2_dx2->base_sizes(1) : TensorShapeRef{};
  if (du1_dx1)
    neml_assert_dbg(u1_base_sizes == du1_dx1->base_sizes(0),
                    "Incompatible base sizes for chain rule");
  if (du2_dx2)
    neml_assert_dbg(u2_base_sizes == du2_dx2->base_sizes(0),
                    "Incompatible base sizes for chain rule");

  // Reshape the derivatives to 2D/3D matrices
  auto d2y_du1u2_f = d2y_du1u2.tensor().base_reshape(
      {utils::numel(y_base_sizes), utils::numel(u1_base_sizes), utils::numel(u2_base_sizes)});
  Tensor du1_dx1_f, du2_dx2_f;
  if (du1_dx1)
    du1_dx1_f =
        du1_dx1->tensor().base_reshape({utils::numel(u1_base_sizes), utils::numel(x1_base_sizes)});
  if (du2_dx2)
    du2_dx2_f =
        du2_dx2->tensor().base_reshape({utils::numel(u2_base_sizes), utils::numel(x2_base_sizes)});

  // // Align dependent intermediate dimensions (for matrix multiplication)
  // d2y_du1u2_f = d2y_du1u2_f.intmd_unsqueeze(-1, Size(x1_intrsc_intmd_dim + x2_intrsc_intmd_dim));
  // if (du1_dx1)
  //   du1_dx1_f =
  //       du1_dx1_f
  //           .intmd_unsqueeze(Size(-u1_intrsc_intmd_dim - x1_intrsc_intmd_dim - 1),
  //           Size(y_intrsc_intmd_dim)) .intmd_unsqueeze(Size(-x1_intrsc_intmd_dim - 1),
  //           Size(u2_intrsc_intmd_dim)) .intmd_unsqueeze(-1, Size(x2_intrsc_intmd_dim));
  // if (du2_dx2)
  //   du2_dx2_f = du2_dx2_f
  //                   .intmd_unsqueeze(Size(-u2_intrsc_intmd_dim - x2_intrsc_intmd_dim - 1),
  //                                    Size(y_intrsc_intmd_dim + u1_intrsc_intmd_dim))
  //                   .intmd_unsqueeze(Size(-x2_intrsc_intmd_dim - 1), Size(x1_intrsc_intmd_dim));

  // Apply chain rule via matrix multiplication
  Tensor d2y_dx1x2_f;
  if (du1_dx1 && du2_dx2)
    d2y_dx1x2_f = neml2::einsum("...ipq,...pj,...qk", {d2y_du1u2_f, du1_dx1_f, du2_dx2_f});
  else if (du1_dx1)
    d2y_dx1x2_f = neml2::einsum("...ipk,...pj", {d2y_du1u2_f, du1_dx1_f});
  else if (du2_dx2)
    d2y_dx1x2_f = neml2::einsum("...ijq,...qk", {d2y_du1u2_f, du2_dx2_f});

  // // Reduce dependent intermediate dimensions for u1 and u2
  // if (u1_intrsc_intmd_dim + u2_intrsc_intmd_dim > 0)
  // {
  //   TensorShape reduce_dims(u1_intrsc_intmd_dim + u2_intrsc_intmd_dim);
  //   std::iota(reduce_dims.begin(),
  //             reduce_dims.end(),
  //             -u1_intrsc_intmd_dim - u2_intrsc_intmd_dim - x1_intrsc_intmd_dim -
  //             x2_intrsc_intmd_dim);
  //   d2y_dx1x2_f = neml2::intmd_sum(d2y_dx1x2_f, reduce_dims, false);
  // }

  // Reshape back to original base sizes
  auto d2y_dx1x2 =
      Derivative<2>({d2y_du1u2.intrsc_intmd_sizes(0),
                     du1_dx1 ? du1_dx1->intrsc_intmd_sizes(1) : d2y_du1u2.intrsc_intmd_sizes(1),
                     du2_dx2 ? du2_dx2->intrsc_intmd_sizes(1) : d2y_du1u2.intrsc_intmd_sizes(2)},
                    {y_base_sizes, x1_base_sizes, x2_base_sizes});
  d2y_dx1x2 =
      d2y_dx1x2_f.base_reshape(utils::add_shapes(y_base_sizes, x1_base_sizes, x2_base_sizes));
  return d2y_dx1x2;
}

Derivative<2>
chain_rule(const Derivative<1> & dy_du, const Derivative<2> & d2u_dx1x2)
{
  // // Dependent intermediate dimensions
  // const auto y_intrsc_intmd_dim = dy_du.intrsc_intmd_sizes(0).size();
  // const auto u_intrsc_intmd_dim = dy_du.intrsc_intmd_sizes(1).size();
  // const auto x1_intrsc_intmd_dim = d2u_dx1x2.intrsc_intmd_sizes(1).size();
  // const auto x2_intrsc_intmd_dim = d2u_dx1x2.intrsc_intmd_sizes(2).size();
  // neml_assert_dbg(u_intrsc_intmd_dim >= d2u_dx1x2.intrsc_intmd_sizes(0).size(),
  //                 "Incompatible dependent intermediate dimensions for chain rule");

  // Base sizes
  const auto y_base_sizes = dy_du.base_sizes(0);
  const auto u_base_sizes = dy_du.base_sizes(1);
  const auto x1_base_sizes = d2u_dx1x2.base_sizes(1);
  const auto x2_base_sizes = d2u_dx1x2.base_sizes(2);
  neml_assert_dbg(u_base_sizes == d2u_dx1x2.base_sizes(0),
                  "Incompatible base sizes for chain rule");

  // Reshape the derivatives to 2D/3D matrices
  auto dy_du_f =
      dy_du.tensor().base_reshape({utils::numel(y_base_sizes), utils::numel(u_base_sizes)});
  auto d2u_dx1x2_f = d2u_dx1x2.tensor().base_reshape(
      {utils::numel(u_base_sizes), utils::numel(x1_base_sizes), utils::numel(x2_base_sizes)});

  // // Align dependent intermediate dimensions (for matrix multiplication)
  // dy_du_f = dy_du_f.intmd_unsqueeze(-1, Size(x1_intrsc_intmd_dim));
  // dy_du_f = dy_du_f.intmd_unsqueeze(-1, Size(x2_intrsc_intmd_dim));
  // d2u_dx1x2_f = d2u_dx1x2_f.intmd_unsqueeze(
  //     Size(-u_intrsc_intmd_dim - x1_intrsc_intmd_dim - x2_intrsc_intmd_dim - 1),
  //     Size(y_intrsc_intmd_dim));

  // Apply chain rule via matrix multiplication
  auto d2y_dx1x2_f = neml2::einsum("...ip,...pjk", {dy_du_f, d2u_dx1x2_f});

  // // Reduce dependent intermediate dimensions for u
  // if (u_intrsc_intmd_dim > 0)
  // {
  //   TensorShape reduce_dims(u_intrsc_intmd_dim);
  //   std::iota(reduce_dims.begin(),
  //             reduce_dims.end(),
  //             -u_intrsc_intmd_dim - x1_intrsc_intmd_dim - x2_intrsc_intmd_dim);
  //   d2y_dx1x2_f = neml2::intmd_sum(d2y_dx1x2_f, reduce_dims, false);
  // }

  // Reshape back to original base sizes
  auto d2y_dx1x2 = Derivative<2>({dy_du.intrsc_intmd_sizes(0),
                                  d2u_dx1x2.intrsc_intmd_sizes(1),
                                  d2u_dx1x2.intrsc_intmd_sizes(2)},
                                 {y_base_sizes, x1_base_sizes, x2_base_sizes});
  d2y_dx1x2 =
      d2y_dx1x2_f.base_reshape(utils::add_shapes(y_base_sizes, x1_base_sizes, x2_base_sizes));
  return d2y_dx1x2;
}

} // namespace neml2
