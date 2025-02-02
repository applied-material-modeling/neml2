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

#include "neml2/misc/math.h"
#include "neml2/misc/error.h"
#include "neml2/tensors/tensors.h"

#include <torch/autograd.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_solve.h>
#include <ATen/ops/linalg_lu_factor.h>
#include <ATen/ops/linalg_lu_solve.h>

namespace neml2::math
{
ConstantTensors::ConstantTensors()
{
  _full_to_mandel_map = torch::tensor({0, 4, 8, 5, 2, 1}, default_integer_tensor_options());

  _mandel_to_full_map =
      torch::tensor({0, 5, 4, 5, 1, 3, 4, 3, 2}, default_integer_tensor_options());

  _full_to_mandel_factor = torch::tensor({1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2});

  _mandel_to_full_factor =
      torch::tensor({1.0, invsqrt2, invsqrt2, invsqrt2, 1.0, invsqrt2, invsqrt2, invsqrt2, 1.0});

  _full_to_skew_map = torch::tensor({7, 2, 3}, default_integer_tensor_options());

  _skew_to_full_map = torch::tensor({0, 2, 1, 2, 0, 0, 1, 0, 0}, default_integer_tensor_options());

  _full_to_skew_factor = torch::tensor({1.0, 1.0, 1.0});

  _skew_to_full_factor = torch::tensor({0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0});
}

ConstantTensors &
ConstantTensors::get()
{
  static ConstantTensors cts;
  return cts;
}

const torch::Tensor &
ConstantTensors::full_to_mandel_map()
{
  return get()._full_to_mandel_map;
}

const torch::Tensor &
ConstantTensors::mandel_to_full_map()
{
  return get()._mandel_to_full_map;
}

const torch::Tensor &
ConstantTensors::full_to_mandel_factor()
{
  return get()._full_to_mandel_factor;
}

const torch::Tensor &
ConstantTensors::mandel_to_full_factor()
{
  return get()._mandel_to_full_factor;
}

const torch::Tensor &
ConstantTensors::full_to_skew_map()
{
  return get()._full_to_skew_map;
}

const torch::Tensor &
ConstantTensors::skew_to_full_map()
{
  return get()._skew_to_full_map;
}

const torch::Tensor &
ConstantTensors::full_to_skew_factor()
{
  return get()._full_to_skew_factor;
}

const torch::Tensor &
ConstantTensors::skew_to_full_factor()
{
  return get()._skew_to_full_factor;
}

Tensor
full_to_reduced(const Tensor & full,
                const torch::Tensor & rmap,
                const torch::Tensor & rfactors,
                Size dim)
{
  auto batch_dim = full.batch_dim();
  auto starting_dim = batch_dim + dim;
  auto trailing_dim = full.dim() - starting_dim - 2; // 2 comes from the reduced axes (3,3)
  auto starting_shape = full.sizes().slice(0, starting_dim);
  auto trailing_shape = full.sizes().slice(starting_dim + 2);

  indexing::TensorIndices net(starting_dim, indexing::None);
  net.push_back(indexing::Ellipsis);
  net.insert(net.end(), trailing_dim, indexing::None);
  auto map =
      rmap.index(net).expand(utils::add_shapes(starting_shape, rmap.sizes()[0], trailing_shape));
  auto factor = rfactors.to(full).index(net);

  return Tensor(
      factor * torch::gather(full.reshape(utils::add_shapes(starting_shape, 9, trailing_shape)),
                             starting_dim,
                             map),
      full.batch_sizes());
}

Tensor
reduced_to_full(const Tensor & reduced,
                const torch::Tensor & rmap,
                const torch::Tensor & rfactors,
                Size dim)
{
  auto batch_dim = reduced.batch_dim();
  auto starting_dim = batch_dim + dim;
  auto trailing_dim = reduced.dim() - starting_dim - 1; // There's only 1 axis to unsqueeze
  auto starting_shape = reduced.sizes().slice(0, starting_dim);
  auto trailing_shape = reduced.sizes().slice(starting_dim + 1);

  indexing::TensorIndices net(starting_dim, indexing::None);
  net.push_back(indexing::Ellipsis);
  net.insert(net.end(), trailing_dim, indexing::None);
  auto map = rmap.index(net).expand(utils::add_shapes(starting_shape, 9, trailing_shape));
  auto factor = rfactors.to(reduced).index(net);

  return Tensor((factor * torch::gather(reduced, starting_dim, map))
                    .reshape(utils::add_shapes(starting_shape, 3, 3, trailing_shape)),
                reduced.batch_sizes());
}

Tensor
full_to_mandel(const Tensor & full, Size dim)
{
  return full_to_reduced(
      full,
      ConstantTensors::full_to_mandel_map().to(full.options().dtype(default_integer_dtype())),
      ConstantTensors::full_to_mandel_factor().to(full.options()),
      dim);
}

Tensor
mandel_to_full(const Tensor & mandel, Size dim)
{
  return reduced_to_full(
      mandel,
      ConstantTensors::mandel_to_full_map().to(mandel.options().dtype(default_integer_dtype())),
      ConstantTensors::mandel_to_full_factor().to(mandel.options()),
      dim);
}

Tensor
full_to_skew(const Tensor & full, Size dim)
{
  return full_to_reduced(
      full,
      ConstantTensors::full_to_skew_map().to(full.options().dtype(default_integer_dtype())),
      ConstantTensors::full_to_skew_factor().to(full.options()),
      dim);
}

Tensor
skew_to_full(const Tensor & skew, Size dim)
{
  return reduced_to_full(
      skew,
      ConstantTensors::skew_to_full_map().to(skew.options().dtype(default_integer_dtype())),
      ConstantTensors::skew_to_full_factor().to(skew.options()),
      dim);
}

std::vector<Tensor>
jacrev(const Tensor & y,
       const std::vector<Tensor> & xs,
       bool retain_graph,
       bool create_graph,
       bool allow_unused)
{
  std::vector<Tensor> dy_dxs(xs.size());

  // Return undefined Tensor if y does not contain any gradient graph
  if (!y.requires_grad())
    return dy_dxs;

  // Check batch shapes
  for (std::size_t i = 0; i < xs.size(); i++)
    neml_assert_dbg(y.batch_sizes() == xs[i].batch_sizes(),
                    "In math::jacrev, the output variable batch shape ",
                    y.batch_sizes(),
                    " is different from the batch shape of x[",
                    i,
                    "] ",
                    xs[i].batch_sizes(),
                    ".");

  const auto opt = y.options().requires_grad(false);

  // Flatten y to handle arbitrarily shaped output
  const auto yf = y.base_flatten();
  const auto G = Scalar::full(1.0, opt).batch_expand(yf.batch_sizes());

  // Initialize derivatives to zero
  std::vector<torch::Tensor> xts(xs.begin(), xs.end());
  std::vector<Tensor> dyf_dxs(xs.size());
  for (std::size_t i = 0; i < xs.size(); i++)
    dyf_dxs[i] = Tensor::zeros(
        yf.batch_sizes(), utils::add_shapes(yf.base_size(0), xs[i].base_sizes()), opt);

  // Use autograd to calculate the derivatives
  for (Size i = 0; i < yf.base_size(0); i++)
  {
    const auto dyfi_dxs = torch::autograd::grad({yf.base_index({i})},
                                                {xts},
                                                {G},
                                                /*retain_graph=*/retain_graph,
                                                /*create_graph=*/create_graph,
                                                /*allow_unused=*/allow_unused);
    neml_assert_dbg(dyfi_dxs.size() == xs.size(),
                    "In math::jacrev, the number of derivatives is ",
                    dyfi_dxs.size(),
                    " but the number of input tensors is ",
                    xs.size(),
                    ".");
    for (std::size_t j = 0; j < xs.size(); j++)
      if (dyfi_dxs[j].defined())
        dyf_dxs[j].base_index_put_({Size(i)}, dyfi_dxs[j]);
  }

  // Reshape the derivative back to the correct shape
  for (std::size_t i = 0; i < xs.size(); i++)
    dy_dxs[i] = dyf_dxs[i].base_reshape(utils::add_shapes(y.base_sizes(), xs[i].base_sizes()));

  return dy_dxs;
}

Tensor
jacrev(const Tensor & y, const Tensor & x, bool retain_graph, bool create_graph, bool allow_unused)
{
  return jacrev(y, std::vector<Tensor>{x}, retain_graph, create_graph, allow_unused)[0];
}

Tensor
base_diag_embed(const Tensor & a, Size offset, Size d1, Size d2)
{
  return Tensor(
      torch::diag_embed(
          a, offset, d1 < 0 ? d1 : d1 + a.batch_dim() + 1, d2 < 0 ? d2 : d2 + a.batch_dim() + 1),
      a.batch_sizes());
}

SR2
skew_and_sym_to_sym(const SR2 & e, const WR2 & w)
{
  // In NEML we used an unrolled form, I don't think I ever found
  // a nice direct notation for this one
  auto E = R2(e);
  auto W = R2(w);
  return SR2(W * E - E * W);
}

SSR4
d_skew_and_sym_to_sym_d_sym(const WR2 & w)
{
  auto I = R2::identity(w.options());
  auto W = R2(w);
  return SSR4(R4(torch::einsum("...ia,...jb->...ijab", {W, I}) -
                 torch::einsum("...ia,...bj->...ijab", {I, W})));
}

SWR4
d_skew_and_sym_to_sym_d_skew(const SR2 & e)
{
  auto I = R2::identity(e.options());
  auto E = R2(e);
  return SWR4(R4(torch::einsum("...ia,...bj->...ijab", {I, E}) -
                 torch::einsum("...ia,...jb->...ijab", {E, I})));
}

WR2
multiply_and_make_skew(const SR2 & a, const SR2 & b)
{
  auto A = R2(a);
  auto B = R2(b);

  return WR2(A * B - B * A);
}

WSR4
d_multiply_and_make_skew_d_first(const SR2 & b)
{
  auto I = R2::identity(b.options());
  auto B = R2(b);
  return WSR4(R4(torch::einsum("...ia,...bj->...ijab", {I, B}) -
                 torch::einsum("...ia,...jb->...ijab", {B, I})));
}

WSR4
d_multiply_and_make_skew_d_second(const SR2 & a)
{
  auto I = R2::identity(a.options());
  auto A = R2(a);
  return WSR4(R4(torch::einsum("...ia,...jb->...ijab", {A, I}) -
                 torch::einsum("...ia,...bj->...ijab", {I, A})));
}

Tensor
base_cat(const std::vector<Tensor> & tensors, Size d)
{
  neml_assert_dbg(!tensors.empty(), "base_cat must be given at least one tensor");
  std::vector<torch::Tensor> torch_tensors(tensors.begin(), tensors.end());
  auto d2 = d < 0 ? d : d + tensors.begin()->batch_dim();
  return neml2::Tensor(torch::cat(torch_tensors, d2), tensors.begin()->batch_sizes());
}

Tensor
base_stack(const std::vector<Tensor> & tensors, Size d)
{
  neml_assert_dbg(!tensors.empty(), "base_stack must be given at least one tensor");
  std::vector<torch::Tensor> torch_tensors(tensors.begin(), tensors.end());
  auto d2 = d < 0 ? d : d + tensors.begin()->batch_dim();
  return neml2::Tensor(torch::stack(torch_tensors, d2), tensors.begin()->batch_sizes());
}

Tensor
base_sum(const Tensor & a, Size d)
{
  auto d2 = d < 0 ? d : d + a.batch_dim();
  return Tensor(torch::sum(a, d2), a.batch_sizes());
}

Tensor
base_mean(const Tensor & a, Size d)
{
  auto d2 = d < 0 ? d : d + a.batch_dim();
  return Tensor(torch::mean(a, d2), a.batch_sizes());
}

Tensor
pow(const Real & a, const Tensor & n)
{
  return Tensor(torch::pow(a, n), n.batch_sizes());
}

Tensor
pow(const Tensor & a, const Tensor & n)
{
  neml_assert_broadcastable_dbg(a, n);
  return Tensor(torch::pow(a, n), broadcast_batch_dim(a, n));
}

namespace linalg
{
Tensor
vector_norm(const Tensor & v)
{
  neml_assert_dbg(v.base_dim() == 0 || v.base_dim() == 1,
                  "v in vector_norm has base dimension ",
                  v.base_dim(),
                  " instead of 0 or 1.");

  // If the vector is a scalar just return its absolute value
  if (v.base_dim() == 0)
    return math::abs(v);

  return Tensor(torch::linalg_vector_norm(
                    v, /*order=*/2, /*dim=*/-1, /*keepdim=*/false, /*dtype=*/c10::nullopt),
                v.batch_sizes());
}

Tensor
inv(const Tensor & m)
{
  return Tensor(torch::linalg_inv(m), m.batch_sizes());
}

Tensor
solve(const Tensor & A, const Tensor & B)
{
  return Tensor(torch::linalg_solve(A, B, /*left=*/true), A.batch_sizes());
}

std::tuple<Tensor, Tensor>
lu_factor(const Tensor & A, bool pivot)
{
  auto [LU, pivots] = torch::linalg_lu_factor(A, pivot);
  return {Tensor(LU, A.batch_sizes()), Tensor(pivots, A.batch_sizes())};
}

Tensor
lu_solve(const Tensor & LU, const Tensor & pivots, const Tensor & B, bool left, bool adjoint)
{
  return Tensor(torch::linalg_lu_solve(LU, pivots, B, left, adjoint), B.batch_sizes());
}
} // namespace linalg
} // namespace neml2
