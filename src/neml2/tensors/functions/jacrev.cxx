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

#include <torch/autograd.h>

#include "neml2/tensors/functions/jacrev.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/from_assembly.h"

namespace neml2
{
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
    neml_assert_dbg(y.dynamic_sizes() == xs[i].dynamic_sizes(),
                    "In jacrev, the output variable dynamic shape ",
                    y.dynamic_sizes(),
                    " is different from the dynamic shape of x[",
                    i,
                    "] ",
                    xs[i].dynamic_sizes(),
                    ".");

  const auto opt = y.options().requires_grad(false);

  // Flatten y to handle arbitrarily shaped output
  const auto yf = y.static_flatten();
  const auto G = Scalar::full(1.0, opt).dynamic_expand(yf.batch_sizes());

  // Initialize derivatives to zero
  std::vector<ATensor> xts(xs.begin(), xs.end());
  std::vector<Tensor> dyf_dxs(xs.size());
  for (std::size_t i = 0; i < xs.size(); i++)
    dyf_dxs[i] = Tensor::zeros(
        yf.dynamic_sizes(), {}, utils::add_shapes(yf.base_size(0), xs[i].static_sizes()), opt);

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
                    "In jacrev, the number of derivatives is ",
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
  {
    dy_dxs[i] = dyf_dxs[i].base_reshape(
        utils::add_shapes(utils::numel(y.static_sizes()), utils::numel(xs[i].static_sizes())));
    dy_dxs[i] = from_assembly<2>(
        dy_dxs[i], {y.intmd_sizes(), xs[i].intmd_sizes()}, {y.base_sizes(), xs[i].base_sizes()});
  }

  return dy_dxs;
}

Tensor
jacrev(const Tensor & y, const Tensor & x, bool retain_graph, bool create_graph, bool allow_unused)
{
  return jacrev(y, std::vector<Tensor>{x}, retain_graph, create_graph, allow_unused)[0];
}
} // namespace neml2
