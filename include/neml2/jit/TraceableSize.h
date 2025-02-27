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

#include <variant>
#include <ATen/core/Tensor.h>
#include "neml2/misc/types.h"

namespace neml2
{
/**
 * @brief Traceable size
 *
 * Similar to neml2::TraceableTensorShape, but only for a single dimension.
 * @see neml2::TraceableTensorShape
 */
struct TraceableSize : public std::variant<Size, ATensor>
{
  using std::variant<Size, ATensor>::variant;

  /// @return a pointer to the ATensor representing the traceable size if it is traceable, otherwise a nullptr
  const ATensor * traceable() const noexcept;

  /// @return the concrete size (without any traceable information)
  Size concrete() const;

  /// @return the size represented as a scalar tensor (possibly traceable)
  ATensor as_tensor() const;
};

/// Comparison operators
///@{
bool operator==(const TraceableSize & lhs, const TraceableSize & rhs);
bool operator!=(const TraceableSize & lhs, const TraceableSize & rhs);
///@}

/// Streaming operator
std::ostream & operator<<(std::ostream & os, const TraceableSize & s);
} // namespace neml2
