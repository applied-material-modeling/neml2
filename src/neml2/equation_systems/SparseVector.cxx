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

#include "neml2/equation_systems/SparseVector.h"
#include "neml2/base/MutableArrayRef.h"
#include "neml2/misc/assertions.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/misc/defaults.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/cat.h"

namespace neml2
{

SparseVector::SparseVector(std::shared_ptr<AxisLayout> layout)
  : tensors(layout->size()),
    layout(std::move(layout))
{
}

SparseVector::SparseVector(std::vector<Tensor> tensors, std::shared_ptr<AxisLayout> layout)
  : tensors(std::move(tensors)),
    layout(std::move(layout))
{
  neml_assert_dbg(tensors.size() == layout->size(), "Number of tensors must match the layout size");
}

std::size_t
SparseVector::size() const
{
  return tensors.size();
}

std::size_t
SparseVector::ngroup() const
{
  return layout->ngroup();
}

SparseVectorView
SparseVector::group(std::size_t i)
{
  neml_assert_dbg(i < ngroup(), "Group index out of range");
  const auto & offsets = layout->offsets;
  auto start = offsets[i];
  auto end = offsets[i + 1];
  return SparseVectorView{MutableArrayRef<Tensor>(tensors.data() + start, end - start),
                          layout->group(i)};
}

std::size_t
SparseVectorView::size() const
{
  return tensors.size();
}

Tensor
SparseVectorView::assemble(bool assemble_intmd) const
{
  // convert to assembly format
  std::vector<Tensor> tf(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i)
  {
    const auto & ti = tensors[i];
    if (!ti.defined())
      continue;
    if (!assemble_intmd)
      tf[i] = ti.base_flatten();
    else
      tf[i] = to_assembly<1>(ti, {layout.intmd_shapes[i]}, {layout.base_shapes[i]});
  }

  // determine tensor options
  auto opt = default_tensor_options();
  for (const auto & t : tf)
    if (t.defined())
    {
      opt = t.options();
      break;
    }

  // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with zeros.
  const auto new_dynamic_sizes = utils::broadcast_dynamic_sizes(tf);
  const auto new_intmd_sizes = utils::broadcast_intmd_sizes(tf);
  for (std::size_t i = 0; i < tf.size(); ++i)
  {
    auto & tfi = tf[i];
    if (tfi.defined())
      tfi = tfi.batch_expand(new_dynamic_sizes, new_intmd_sizes);
    else
    {
      auto s = utils::numel(layout.base_shapes[i]);
      if (assemble_intmd)
        s *= utils::numel(layout.intmd_shapes[i]);
      tfi = Tensor::zeros(new_dynamic_sizes, new_intmd_sizes, s, opt);
    }
  }

  return base_cat(tf, -1);
}

void
SparseVectorView::disassemble(const Tensor & t, bool assemble_intmd)
{
  neml_assert_dbg(t.base_dim() == 1, "disassemble expects base dimension of 1, got ", t.base_dim());
  if (assemble_intmd)
    neml_assert_dbg(t.intmd_dim() == 0,
                    "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                    t.intmd_dim());

  const auto ss = layout.storage_sizes(assemble_intmd);
  const auto & D = t.dynamic_sizes();
  const auto I = t.intmd_dim();
  const auto vs = t.split(ss, -1);
  const auto n = vs.size();

  for (std::size_t i = 0; i < n; ++i)
  {
    auto ti = Tensor(vs[i], D, I);
    if (!assemble_intmd)
      tensors[i] = ti.base_reshape(layout.base_shapes[i]);
    else
      tensors[i] = from_assembly<1>(ti, {layout.intmd_shapes[i]}, {layout.base_shapes[i]});
  }
}

} // namespace neml2
