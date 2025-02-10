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

#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
#define DEFINE_CAT(T)                                                                              \
  T batch_cat(const std::vector<T> & tensors, Size d)                                              \
  {                                                                                                \
    neml_assert_dbg(!tensors.empty(), "batch_cat must be given at least one tensor");              \
    std::vector<ATensor> torch_tensors(tensors.begin(), tensors.end());                            \
    auto d2 = d >= 0 ? d : d - tensors.begin()->base_dim();                                        \
    return T(at::cat(torch_tensors, d2), tensors.begin()->batch_dim());                            \
  }                                                                                                \
  T batch_cat(const std::initializer_list<T> & tensors, Size d)                                    \
  {                                                                                                \
    return batch_cat(std::vector<T>(tensors), d);                                                  \
  }                                                                                                \
  static_assert(true)
FOR_ALL_TENSORBASE(DEFINE_CAT);

Tensor
base_cat(const std::vector<Tensor> & tensors, Size d)
{
  neml_assert_dbg(!tensors.empty(), "base_cat must be given at least one tensor");
  std::vector<ATensor> torch_tensors(tensors.begin(), tensors.end());
  auto d2 = d < 0 ? d : d + tensors.begin()->batch_dim();
  return neml2::Tensor(at::cat(torch_tensors, d2), tensors.begin()->batch_sizes());
}
} // namespace neml2
