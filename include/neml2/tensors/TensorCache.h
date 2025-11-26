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

#pragma once

#include "neml2/tensors/Tensor.h"

namespace neml2
{
struct TensorCache
{
public:
  /**
   * @brief Construct a new Tensor Cache object
   *
   * The constructor takes a lambda that creates a tensor given a set of tensor options. The lambda
   * is invoked upon a cache miss.
   */
  TensorCache(std::function<Tensor(const TensorOptions &)> &&);

  /// Get the tensor with the given tensor options. If the tensor does not exist in the cache, it is created.
  const Tensor & operator()(const TensorOptions &);

private:
  struct CacheKey
  {
    Dtype dtype = kFloat64;
    Device device;
    bool operator==(const CacheKey & other) const;
  };

  /// Lambda that creates a tensor given a set of tensor options
  std::function<Tensor(const TensorOptions &)> _creator;

  /// Keys of cached tensors
  std::vector<CacheKey> _cached_keys;

  /// Cache of already created tensors
  std::vector<Tensor> _cached_tensors;
};
} // namespace neml2
