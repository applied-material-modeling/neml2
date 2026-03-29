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

#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/misc/assertions.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/equation_systems/SparseVector.h"

namespace neml2
{

AssembledVector::AssembledVector(AxisLayout l)
  : tensors(l.ngroup()),
    layout(std::move(l))
{
}

AssembledVector::AssembledVector(AxisLayout l, std::vector<Tensor> t)
  : tensors(std::move(t)),
    layout(std::move(l))
{
  neml_assert_dbg(tensors.size() == layout.ngroup(),
                  "Number of tensors must match the layout group size");
}

AssembledVector
AssembledVector::group(std::size_t i) const
{
  auto [start, end] = layout.group_offsets(i);
  std::vector<Tensor> ts(tensors.begin() + Size(start), tensors.begin() + Size(end));
  return AssembledVector(layout.group(i), std::move(ts));
}

SparseVector
AssembledVector::disassemble() const
{
  std::vector<Tensor> sp_tensors(layout.nvar());

  for (std::size_t grp = 0; grp < layout.ngroup(); ++grp)
  {
    const auto & t = tensors[grp];
    const auto [istart, iend] = layout.group_offsets(grp);
    const auto istr = layout.group_istr(grp);
    const bool assemble_intmd = (istr == AxisLayout::IStructure::DENSE);

    neml_assert_dbg(
        t.base_dim() == 1, "disassemble expects base dimension of 1, got ", t.base_dim());
    if (assemble_intmd)
      neml_assert_dbg(t.intmd_dim() == 0,
                      "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                      t.intmd_dim());

    const auto ss = layout.group(grp).storage_sizes(assemble_intmd);
    const auto & D = t.dynamic_sizes();
    const auto I = t.intmd_dim();
    const auto vs = t.split(ss, -1);

    for (std::size_t i = istart; i < iend; ++i)
    {
      auto ti = Tensor(vs[i - istart], D, I);
      if (!assemble_intmd)
        sp_tensors[i] = ti.base_reshape(layout.base_sizes(i));
      else
        sp_tensors[i] = from_assembly<1>(ti, {layout.intmd_sizes(i)}, {layout.base_sizes(i)});
    }
  }

  return SparseVector(layout, std::move(sp_tensors));
}

} // namespace neml2
