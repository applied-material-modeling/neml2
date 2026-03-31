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
#include "neml2/misc/assertions.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/misc/defaults.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/equation_systems/AssembledVector.h"

namespace neml2
{

SparseVector::SparseVector(AxisLayout l)
  : tensors(l.nvar()),
    layout(std::move(l))
{
}

SparseVector::SparseVector(AxisLayout l, std::vector<Tensor> t)
  : tensors(std::move(t)),
    layout(std::move(l))
{
  neml_assert_dbg(tensors.size() == layout.nvar(), "Number of tensors must match the layout size");
}

TensorOptions
SparseVector::options() const
{
  for (const auto & t : tensors)
    if (t.defined())
      return t.options();
  return default_tensor_options();
}

SparseVector
SparseVector::group(std::size_t i) const
{
  auto [start, end] = layout.group_offsets(i);
  std::vector<Tensor> ts(tensors.begin() + Size(start), tensors.begin() + Size(end));
  return SparseVector(layout.group(i), std::move(ts));
}

AssembledVector
SparseVector::assemble() const
{
  std::vector<Tensor> asm_tensors(layout.ngroup());

  for (std::size_t grp = 0; grp < layout.ngroup(); ++grp)
  {
    const auto glayout = layout.group(grp);
    const auto [istart, iend] = layout.group_offsets(grp);
    const auto istr = layout.istr(grp);
    const bool assemble_intmd = (istr == AxisLayout::IStructure::DENSE);

    // convert to assembly format
    std::vector<Tensor> tf(iend - istart);
    for (std::size_t i = 0; i < iend - istart; ++i)
    {
      const auto & ti = tensors[istart + i];
      if (!ti.defined())
        continue;
      if (!assemble_intmd)
        tf[i] = ti.base_flatten();
      else
        tf[i] = to_assembly<1>(ti, {glayout.intmd_sizes(i)}, {glayout.base_sizes(i)});
    }

    // determine tensor options
    auto opt = default_tensor_options();
    for (const auto & t : tf)
      if (t.defined())
      {
        opt = t.options();
        break;
      }

    // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with
    // zeros.
    const auto ss = glayout.storage_sizes(assemble_intmd);
    const auto new_dynamic_sizes = utils::broadcast_dynamic_sizes(tf);
    const auto new_intmd_sizes = utils::broadcast_intmd_sizes(tf);
    for (std::size_t i = 0; i < tf.size(); ++i)
    {
      auto & tfi = tf[i];
      if (tfi.defined())
        tfi = tfi.batch_expand(new_dynamic_sizes, new_intmd_sizes);
      else
        tfi = Tensor::zeros(new_dynamic_sizes, new_intmd_sizes, ss[i], opt);
    }

    asm_tensors[grp] = base_cat(tf, -1);
  }

  return AssembledVector(layout, std::move(asm_tensors));
}

} // namespace neml2
