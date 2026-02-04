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

#include <torch/csrc/jit/frontend/tracer.h>

#include "neml2/tensors/functions/sum_to_size.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/jit.h"

namespace neml2
{
#define DEFINE_SUM_TO_SIZE(T)                                                                      \
  T dynamic_sum_to_size(const T & a, const TraceableTensorShape & shape)                           \
  {                                                                                                \
    neml_assert_dbg(at::is_expandable_to(shape.concrete(), a.dynamic_sizes().concrete()),          \
                    "Cannot sum tensor of dynamic shape ",                                         \
                    a.dynamic_sizes(),                                                             \
                    " to shape ",                                                                  \
                    shape,                                                                         \
                    ".");                                                                          \
                                                                                                   \
    if (jit::tracer::isTracing())                                                                  \
      for (std::size_t i = 0; i < shape.size(); ++i)                                               \
        if (const auto * const si = shape[i].traceable())                                          \
          jit::tracer::ArgumentStash::stashIntArrayRefElem(                                        \
              "size", shape.size() + a.static_dim(), i, *si);                                      \
                                                                                                   \
    return T(a.sum_to_size(utils::add_shapes(shape.concrete(), a.static_sizes())),                 \
             shape,                                                                                \
             a.intmd_dim());                                                                       \
  }                                                                                                \
                                                                                                   \
  T intmd_sum_to_size(const T & a, TensorShapeRef shape)                                           \
  {                                                                                                \
    neml_assert_dbg(at::is_expandable_to(shape, a.intmd_sizes()),                                  \
                    "Cannot sum tensor of intermediate shape ",                                    \
                    a.intmd_sizes(),                                                               \
                    " to shape ",                                                                  \
                    shape,                                                                         \
                    ".");                                                                          \
    neml_assert_dbg(Size(shape.size()) == a.intmd_dim(),                                           \
                    "Cannot sum tensor of intermediate shape ",                                    \
                    a.intmd_sizes(),                                                               \
                    " to shape ",                                                                  \
                    shape,                                                                         \
                    " with different number of intermediate dimensions (",                         \
                    a.intmd_dim(),                                                                 \
                    ").");                                                                         \
                                                                                                   \
    if (a.intmd_sizes() == shape)                                                                  \
      return a;                                                                                    \
                                                                                                   \
    if (jit::tracer::isTracing())                                                                  \
      for (Size i = 0; i < a.dynamic_dim(); ++i)                                                   \
        if (const auto * const si = a.dynamic_size(i).traceable())                                 \
          jit::tracer::ArgumentStash::stashIntArrayRefElem("size", a.dim(), i, *si);               \
                                                                                                   \
    return T(                                                                                      \
        a.sum_to_size(utils::add_shapes(a.dynamic_sizes().concrete(), shape, a.base_sizes())),     \
        a.dynamic_sizes(),                                                                         \
        a.intmd_dim());                                                                            \
  }                                                                                                \
  static_assert(true)
FOR_ALL_TENSORBASE(DEFINE_SUM_TO_SIZE);

Tensor
base_sum_to_size(const Tensor & a, TensorShapeRef shape)
{
  neml_assert_dbg(at::is_expandable_to(shape, a.base_sizes()),
                  "Cannot sum tensor of base shape ",
                  a.base_sizes(),
                  " to shape ",
                  shape,
                  ".");
  neml_assert_dbg(Size(shape.size()) == a.base_dim(),
                  "Cannot sum tensor of base shape ",
                  a.base_sizes(),
                  " to shape ",
                  shape,
                  " with different number of base dimensions (",
                  a.base_dim(),
                  ").");

  if (a.base_sizes() == shape)
    return a;

  // Record the dynamic sizes in the traced graph if we are tracing
  if (jit::tracer::isTracing())
    for (Size i = 0; i < a.dynamic_dim(); ++i)
      if (const auto * const si = a.dynamic_size(i).traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem("size", a.dim(), i, *si);

  return Tensor(
      a.sum_to_size(utils::add_shapes(a.dynamic_sizes().concrete(), a.intmd_sizes(), shape)),
      a.dynamic_sizes(),
      a.intmd_dim());
}
} // namespace neml2
