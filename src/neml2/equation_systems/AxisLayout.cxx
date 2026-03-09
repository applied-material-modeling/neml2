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

#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{

AxisLayout::AxisLayout(const std::vector<std::vector<LabeledAxisAccessor>> & vars,
                       std::vector<TensorShape> intmd_shapes,
                       std::vector<TensorShape> base_shapes)
  : intmd_shapes(std::move(intmd_shapes)),
    base_shapes(std::move(base_shapes))
{
  offsets.push_back(0);
  for (const auto & grp : vars)
  {
    offsets.push_back(offsets.back() + grp.size());
    this->vars.insert(this->vars.end(), grp.begin(), grp.end());
  }
  neml_assert_dbg(this->vars.size() == this->intmd_shapes.size(),
                  "Number of variables must match the number of intermediate shapes");
  neml_assert_dbg(this->vars.size() == this->base_shapes.size(),
                  "Number of variables must match the number of base shapes");
}

std::size_t
AxisLayout::size() const
{
  return vars.size();
}

std::size_t
AxisLayout::ngroup() const
{
  return offsets.size() - 1;
}

AxisLayoutView
AxisLayout::group(std::size_t idx) const
{
  neml_assert_dbg(idx < ngroup(), "Group index out of range");
  auto start = offsets[idx];
  auto end = offsets[idx + 1];
  return AxisLayoutView{ArrayRef<LabeledAxisAccessor>(vars.data() + start, end - start),
                        ArrayRef<TensorShape>(intmd_shapes.data() + start, end - start),
                        ArrayRef<TensorShape>(base_shapes.data() + start, end - start)};
}

std::size_t
AxisLayoutView::size() const
{
  return vars.size();
}

std::vector<Size>
AxisLayoutView::storage_sizes(bool include_intmd) const
{
  std::vector<Size> ss(vars.size());
  for (std::size_t i = 0; i < vars.size(); ++i)
  {
    auto s = utils::numel(base_shapes[i]);
    if (include_intmd)
      s *= utils::numel(intmd_shapes[i]);
    ss[i] = s;
  }
  return ss;
}

} // namespace neml2
