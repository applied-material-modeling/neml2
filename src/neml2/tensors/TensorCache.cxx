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

#include "neml2/tensors/TensorCache.h"
#include <c10/core/ScalarTypeToTypeMeta.h>

namespace neml2
{
TensorCache::TensorCache(std::function<Tensor(const TensorOptions &)> && creator)
  : _creator(std::move(creator))
{
}

const Tensor &
TensorCache::operator()(const TensorOptions & opt)
{
  const auto key = CacheKey{c10::typeMetaToScalarType(opt.dtype()), opt.device()};
  const auto ti = std::find(_cached_keys.begin(), _cached_keys.end(), key);
  if (ti == _cached_keys.end())
  {
    _cached_keys.push_back(key);
    _cached_tensors.emplace_back(_creator(opt));
    return _cached_tensors.back();
  }
  return _cached_tensors[ti - _cached_keys.begin()];
}

bool
TensorCache::CacheKey::operator==(const CacheKey & other) const
{
  return dtype == other.dtype && device == other.device;
}
}
