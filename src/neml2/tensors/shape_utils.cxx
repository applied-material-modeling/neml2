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

#include <numeric>

#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2::utils
{
Size
normalize_dim(Size d, Size dl, Size du)
{
  auto delta = du - dl;
  neml_assert_dbg(d >= -delta && d < delta,
                  "The dimension ",
                  d,
                  " is out of range [",
                  -delta,
                  ", ",
                  delta,
                  ").");
  return d >= 0 ? dl + d : du + d;
}

TensorShape
normalize_dims(ArrayRef<Size> d, Size dl, Size du)
{
  TensorShape dn;
  dn.reserve(d.size());
  std::transform(d.begin(),
                 d.end(),
                 std::back_inserter(dn),
                 [dl, du](Size dim) { return normalize_dim(dim, dl, du); });
  return dn;
}

Size
normalize_itr(Size d, Size dl, Size du)
{
  auto delta = du - dl;
  neml_assert_dbg(d >= -delta - 1 && d <= delta,
                  "The dimension ",
                  d,
                  " is out of range [",
                  -delta - 1,
                  ", ",
                  delta,
                  "].");
  return d >= 0 ? dl + d : du + d + 1;
}

TensorShape
normalize_itrs(ArrayRef<Size> d, Size dl, Size du)
{
  TensorShape dn;
  dn.reserve(d.size());
  std::transform(d.begin(),
                 d.end(),
                 std::back_inserter(dn),
                 [dl, du](Size dim) { return normalize_itr(dim, dl, du); });
  return dn;
}

Size
numel(TensorShapeRef shape)
{
  Size sz = 1;
  return std::accumulate(shape.begin(), shape.end(), sz, std::multiplies<>());
}

TensorShape
pad_prepend(TensorShapeRef s, std::size_t dim, Size pad)
{
  neml_assert(s.size() <= dim, "pad_prepend given shape ", s, " and dim ", dim);
  TensorShape s2(s);
  s2.insert(s2.begin(), dim - s.size(), pad);
  return s2;
}

std::vector<TensorShape>
shape_refs_to_shapes(const std::vector<TensorShapeRef> & shape_refs)
{
  std::vector<TensorShape> shapes;
  shapes.reserve(shape_refs.size());
  for (const auto & sr : shape_refs)
    shapes.emplace_back(sr);
  return shapes;
}

std::vector<TensorShapeRef>
shapes_to_shape_refs(const std::vector<TensorShape> & shapes)
{
  std::vector<TensorShapeRef> shape_refs;
  shape_refs.reserve(shapes.size());
  for (const auto & s : shapes)
    shape_refs.emplace_back(s);
  return shape_refs;
}

} // namespace neml2::utils
