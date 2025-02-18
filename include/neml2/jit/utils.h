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

#include "neml2/misc/types.h"
#include "neml2/misc/errors.h"
#include "neml2/jit/TraceableTensorShape.h"

namespace neml2
{
/// Assert that we are currently tracing
void neml_assert_tracing();

/// Assert that we are currently NOT tracing
void neml_assert_not_tracing();

/// Assert that we are currently tracing (only effective in debug mode)
void neml_assert_tracing_dbg();

/// Assert that we are currently NOT tracing (only effective in debug mode)
void neml_assert_not_tracing_dbg();

namespace utils
{
/// @brief Extract the batch shape of a tensor given batch dimension
/// The extracted batch shape will be _traceable_. @see neml2::TraceableTensorShape
TraceableTensorShape extract_batch_sizes(const ATensor & tensor, Size batch_dim);

template <typename... S>
TraceableTensorShape add_traceable_shapes(const S &... shape);

/// Print last evaluated optimized graph
std::shared_ptr<jit::Graph> last_executed_optimized_graph();

namespace details
{
template <typename... S>
TraceableTensorShape
add_traceable_shapes_impl(TraceableTensorShape &, const TraceableTensorShape &, const S &...);
} // namespace details
} // namespace utils
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

namespace neml2::utils
{
template <typename... S>
TraceableTensorShape
add_traceable_shapes(const S &... shape)
{
  TraceableTensorShape net;
  return details::add_traceable_shapes_impl(net, shape...);
}

namespace details
{
template <typename... S>
TraceableTensorShape
add_traceable_shapes_impl(TraceableTensorShape & net,
                          const TraceableTensorShape & s,
                          const S &... rest)
{
  net.insert(net.end(), s.begin(), s.end());

  if constexpr (sizeof...(rest) == 0)
    return std::move(net);
  else
    return add_traceable_shapes_impl(net, rest...);
}
} // namespace details
} // namespace neml2::utils
