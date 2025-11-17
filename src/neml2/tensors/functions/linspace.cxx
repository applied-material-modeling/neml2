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

#include "neml2/tensors/functions/linspace.h"
#include "neml2/tensors/tensors.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
#define DEFINE_LINSPACE(T)                                                                         \
  T dynamic_linspace(const T & start, const T & end, Size nstep, Size dim)                         \
  {                                                                                                \
    neml_assert_dbg(nstep > 0, "nstep must be positive.");                                         \
                                                                                                   \
    if (nstep > 1)                                                                                 \
    {                                                                                              \
      auto delta = end - start;                                                                    \
      Scalar steps(at::arange(nstep, delta.options()) / (nstep - 1), 0);                           \
      auto res = start.dynamic_unsqueeze(-1) + steps * delta.dynamic_unsqueeze(-1);                \
      return res.dynamic_movedim(-1, dim);                                                         \
    }                                                                                              \
                                                                                                   \
    auto B = utils::broadcast_dynamic_sizes({start, end});                                         \
    return start.dynamic_expand(B).dynamic_unsqueeze(dim);                                         \
  }                                                                                                \
                                                                                                   \
  T intmd_linspace(const T & start, const T & end, Size nstep, Size dim)                           \
  {                                                                                                \
    neml_assert_dbg(nstep > 0, "nstep must be positive.");                                         \
                                                                                                   \
    if (nstep > 1)                                                                                 \
    {                                                                                              \
      auto delta = end - start;                                                                    \
      Scalar steps(at::arange(nstep, delta.options()) / (nstep - 1), 1);                           \
      auto res = start.intmd_unsqueeze(-1) + steps * delta.intmd_unsqueeze(-1);                    \
      return res.intmd_movedim(-1, dim);                                                           \
    }                                                                                              \
                                                                                                   \
    auto B = utils::broadcast_sizes(start.intmd_sizes(), end.intmd_sizes());                       \
    return start.intmd_expand(B).intmd_unsqueeze(dim);                                             \
  }                                                                                                \
  static_assert(true)
FOR_ALL_TENSORBASE(DEFINE_LINSPACE);

Tensor
base_linspace(const Tensor & start, const Tensor & end, Size nstep, Size dim)
{
  neml_assert_dbg(nstep > 0, "nstep must be positive.");

  if (nstep > 1)
  {
    auto delta = end - start;
    Tensor steps(at::arange(nstep, delta.options()) / (nstep - 1), 0, 0);
    auto res = start.base_unsqueeze(0) + steps * delta.base_unsqueeze(0);
    return res.base_movedim(0, dim);
  }

  auto B = utils::broadcast_sizes(start.base_sizes(), end.base_sizes());
  return start.base_expand(B).base_unsqueeze(dim);
}
} // namespace neml2
