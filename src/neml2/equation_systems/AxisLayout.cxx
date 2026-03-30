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
                       std::vector<TensorShape> base_shapes,
                       std::vector<IStructure> istrs)
  : _intmd_shapes(std::move(intmd_shapes)),
    _base_shapes(std::move(base_shapes)),
    _istrs(std::move(istrs))
{
  _offsets.push_back(0);
  for (const auto & grp : vars)
  {
    _offsets.push_back(_offsets.back() + grp.size());
    _vars.insert(_vars.end(), grp.begin(), grp.end());
    _end += grp.size();
  }
  neml_assert_dbg(_vars.size() == _intmd_shapes.size(),
                  "Number of variables must match the number of intermediate shapes");
  neml_assert_dbg(_vars.size() == _base_shapes.size(),
                  "Number of variables must match the number of base shapes");
  neml_assert_dbg(ngroup() == _istrs.size(),
                  "Number of variable groups must match the number of IStructure entries");
}

AxisLayout::AxisLayout(const AxisLayout * parent,
                       std::size_t start,
                       std::size_t end,
                       std::vector<std::size_t> offsets)
  : _offsets(std::move(offsets)),
    _parent(parent),
    _start(start),
    _end(end)
{
  neml_assert_dbg(start <= end && end <= parent->nvar(), "Invalid view range");
}

std::size_t
AxisLayout::ngroup() const
{
  if (_offsets.empty())
    return 1;
  return _offsets.size() - 1;
}

std::pair<std::size_t, std::size_t>
AxisLayout::group_offsets(std::size_t idx) const
{
  neml_assert(!_offsets.empty(), "Cannot call group_offsets() on a sub-group view");
  neml_assert(idx < ngroup(), "Group index out of range");
  return {_offsets[idx], _offsets[idx + 1]};
}

AxisLayout
AxisLayout::group(std::size_t idx) const
{
  auto [start, end] = group_offsets(idx);
  return AxisLayout(_parent ? _parent : this, start, end);
}

AxisLayout::IStructure
AxisLayout::istr(std::size_t idx) const
{
  neml_assert(!_offsets.empty(), "Cannot call istr() on a sub-group view");
  neml_assert(idx < ngroup(), "Group index out of range");
  return _parent ? _parent->istr(idx) : _istrs[idx];
}

AxisLayout
AxisLayout::view() const
{
  return AxisLayout(_parent ? _parent : this, _start, _end, _offsets);
}

std::size_t
AxisLayout::nvar() const
{
  return _parent ? _end - _start : _vars.size();
}

std::vector<Size>
AxisLayout::storage_sizes(bool include_intmd) const
{
  std::vector<Size> ss(nvar());
  for (std::size_t i = 0; i < nvar(); ++i)
  {
    auto s = utils::numel(base_sizes(i));
    if (include_intmd)
      s *= utils::numel(intmd_sizes(i));
    ss[i] = s;
  }
  return ss;
}

const LabeledAxisAccessor &
AxisLayout::var(std::size_t idx) const
{
  neml_assert(idx < nvar(), "Variable index out of range");
  return _parent ? _parent->var(_start + idx) : _vars[idx];
}

const TensorShape &
AxisLayout::intmd_sizes(std::size_t idx) const
{
  neml_assert(idx < nvar(), "Variable index out of range");
  return _parent ? _parent->intmd_sizes(_start + idx) : _intmd_shapes[idx];
}

const TensorShape &
AxisLayout::base_sizes(std::size_t idx) const
{
  neml_assert(idx < nvar(), "Variable index out of range");
  return _parent ? _parent->base_sizes(_start + idx) : _base_shapes[idx];
}

} // namespace neml2
