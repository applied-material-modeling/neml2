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

#include "neml2/tensors/jit.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
void
neml_assert_tracing()
{
  neml_assert(jit::tracer::isTracing(), "Expected to be tracing but not tracing");
}

void
neml_assert_not_tracing()
{
  neml_assert(!jit::tracer::isTracing(), "Tracing is prohibited in this region");
}

void
neml_assert_tracing_dbg()
{
#ifndef NDEBUG
  neml_assert_tracing();
#endif
}

void
neml_assert_not_tracing_dbg()
{
#ifndef NDEBUG
  neml_assert_not_tracing();
#endif
}

namespace utils
{
TraceableSize
traceable_numel(const TraceableTensorShape & shape)
{
  TraceableSize sz = 1;
  return std::accumulate(shape.begin(), shape.end(), sz, std::multiplies<>());
}

TraceableTensorShape
extract_traceable_sizes(const ATensor & tensor, std::size_t n, std::size_t m)
{
  // It's easy if we are not tracing
  if (!jit::tracer::isTracing())
    return tensor.sizes().slice(n, m);

  // Put the sizes into the traced graph if we are tracing
  // TODO: This could be optimized
  TraceableTensorShape sizes;
  for (std::size_t i = n; i < n + m; ++i)
    sizes.emplace_back(jit::tracer::getSizeOf(tensor, Size(i)));
  return sizes;
}

std::shared_ptr<jit::Graph>
last_executed_optimized_graph()
{
  return jit::lastExecutedOptimizedGraph();
}
} // namespace utils
} // namespace neml2
