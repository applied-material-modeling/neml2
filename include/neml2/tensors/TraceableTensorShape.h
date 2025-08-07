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

#include "neml2/tensors/TraceableSize.h"

namespace neml2
{
/**
 * @brief Traceable tensor shape
 *
 * A tensor shape can be either a concrete shape or a traceable tensor. This is useful when we need
 * to trace a function graph and let it generalize to other batch shapes.
 */
struct TraceableTensorShape : public SmallVector<TraceableSize, 8>
{
  using SmallVector<TraceableSize, 8>::SmallVector;
  using Size = int64_t;

  TraceableTensorShape(const TensorShape & shape);
  TraceableTensorShape(TensorShapeRef shape);
  TraceableTensorShape(Size shape);
  TraceableTensorShape(const ATensor & shape);

  /// Slice the shape
  TraceableTensorShape slice(std::size_t N, std::size_t M) const;

  /// Chop-off the first N elements of the shape, semantically the same as ArrayRef::slice, but traceable.
  TraceableTensorShape slice(std::size_t N) const;

  /// @return the concrete shape (without any traceable information)
  TensorShape concrete() const;

  /// @return the shape represented as a scalar tensor (possibly traceable)
  ATensor as_tensor() const;
};

/// Comparison operators
///@{
bool operator==(const TraceableTensorShape & lhs, const TraceableTensorShape & rhs);
bool operator!=(const TraceableTensorShape & lhs, const TraceableTensorShape & rhs);
///@}
} // namespace neml2
