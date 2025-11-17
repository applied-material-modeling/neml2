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

#include "neml2/dispatchers/ValueMapLoader.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
static std::size_t
broadcast_dynamic_size(const ValueMap & value_map, Size d)
{
  Size size = 0;

  for (auto && [key, tensor] : value_map)
    if (d >= -tensor.dynamic_dim() && d < tensor.dynamic_dim())
      size = std::max(size, tensor.dynamic_size(d).concrete());

  for (auto && [key, tensor] : value_map)
    if (d >= -tensor.dynamic_dim() && d < tensor.dynamic_dim())
    {
      auto s = tensor.dynamic_size(d).concrete();
      neml_assert(s == 1 || s == size,
                  "Sizes along dynamic dimension ",
                  d,
                  " are not compatible. Expected 1 or ",
                  size,
                  ", got ",
                  s,
                  " for variable ",
                  key);
    }
  return size;
}

ValueMapLoader::ValueMapLoader(const ValueMap & value_map, Size dynamic_dim)
  : _value_map(value_map),
    _dynamic_dim(dynamic_dim),
    _slice_gen(0, broadcast_dynamic_size(value_map, dynamic_dim))
{
}

std::size_t
ValueMapLoader::total() const
{
  return _slice_gen.total();
}

std::pair<std::size_t, ValueMap>
ValueMapLoader::generate(std::size_t n)
{
  auto && [m, slice] = _slice_gen.next(n);

  ValueMap work;
  for (auto && [key, tensor] : _value_map)
  {
    if (_dynamic_dim >= -tensor.dynamic_dim() && _dynamic_dim < tensor.dynamic_dim())
    {
      if (tensor.dynamic_size(_dynamic_dim).concrete() != 1)
        work[key] = tensor.dynamic_slice(_dynamic_dim, slice);
      else
        work[key] = tensor;
    }
    else
      work[key] = tensor;
  }

  return {m, std::move(work)};
}
} // namespace neml2
