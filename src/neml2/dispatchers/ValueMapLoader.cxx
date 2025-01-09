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
std::size_t
broadcast_batch_size(const ValueMap & value_map, Size batch_dim)
{
  Size size = 0;
  for (auto && [key, tensor] : value_map)
    size = std::max(size, tensor.batch_size(batch_dim).concrete());
  for (auto && [key, tensor] : value_map)
  {
    auto s = tensor.batch_size(batch_dim).concrete();
    neml_assert(s == 1 || s == size,
                "Batch sizes along batch dimension ",
                batch_dim,
                " are not compatible. Expected 1 or ",
                size,
                ", got ",
                s,
                ".");
  }
  return size;
}

ValueMapLoader::ValueMapLoader(const ValueMap & value_map, Size batch_dim)
  : _value_map(value_map),
    _batch_dim(batch_dim),
    _slice_gen(0, broadcast_batch_size(value_map, batch_dim))
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
    work[key] = tensor.size(_batch_dim) == 1 ? tensor : tensor.batch_slice(_batch_dim, slice);

  return {m, std::move(work)};
}
} // namespace neml2
