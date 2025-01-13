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
#include "neml2/misc/defaults.h"

namespace neml2
{
class Tensor;

/// Empty tensor like another, i.e. same batch and base shapes, same tensor options, etc.
[[nodiscard]] Tensor empty_like(const Tensor & other);
/// Zero tensor like another, i.e. same batch and base shapes, same tensor options, etc.
[[nodiscard]] Tensor zeros_like(const Tensor & other);
/// Unit tensor like another, i.e. same batch and base shapes, same tensor options, etc.
[[nodiscard]] Tensor ones_like(const Tensor & other);
/// Full tensor like another, i.e. same batch and base shapes, same tensor options, etc.,
/// but filled with a different value
[[nodiscard]] Tensor full_like(const Tensor & other, const LScalar & init);

/// Unbatched empty tensor given base shape
[[nodiscard]] Tensor empty(TensorShapeRef base_shape,
                           const TensorOptions & options = default_tensor_options());
/// Empty tensor given batch and base shapes
[[nodiscard]] Tensor empty(const TraceableTensorShape & batch_shape,
                           TensorShapeRef base_shape,
                           const TensorOptions & options = default_tensor_options());
/// Unbatched tensor filled with zeros given base shape
[[nodiscard]] Tensor zeros(TensorShapeRef base_shape,
                           const TensorOptions & options = default_tensor_options());
/// Zero tensor given batch and base shapes
[[nodiscard]] Tensor zeros(const TraceableTensorShape & batch_shape,
                           TensorShapeRef base_shape,
                           const TensorOptions & options = default_tensor_options());
/// Unbatched tensor filled with ones given base shape
[[nodiscard]] Tensor ones(TensorShapeRef base_shape,
                          const TensorOptions & options = default_tensor_options());
/// Unit tensor given batch and base shapes
[[nodiscard]] Tensor ones(const TraceableTensorShape & batch_shape,
                          TensorShapeRef base_shape,
                          const TensorOptions & options = default_tensor_options());
/// Unbatched tensor filled with a given value given base shape
[[nodiscard]] Tensor full(TensorShapeRef base_shape,
                          const LScalar & init,
                          const TensorOptions & options = default_tensor_options());
/// Full tensor given batch and base shapes
[[nodiscard]] Tensor full(const TraceableTensorShape & batch_shape,
                          TensorShapeRef base_shape,
                          const LScalar & init,
                          const TensorOptions & options = default_tensor_options());
/// Unbatched identity tensor
[[nodiscard]] Tensor identity(Size n, const TensorOptions & options = default_tensor_options());
/// Identity tensor given batch shape and base length
[[nodiscard]] Tensor identity(const TraceableTensorShape & batch_shape,
                              Size n,
                              const TensorOptions & options = default_tensor_options());

/**
 * @brief Create a new tensor by adding a new batch dimension with linear spacing between \p
 * start and \p end.
 *
 * \p start and \p end must be broadcastable. The new batch dimension will be added at the
 * user-specified dimension \p dim which defaults to 0.
 *
 * For example, if \p start has shape `(3, 2; 5, 5)` and \p end has shape `(3, 1; 5, 5)`, then
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cpp
 * linspace(start, end, 100, 1);
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * will have shape `(3, 100, 2; 5, 5)`, note the location of the new dimension and the
 * broadcasting.
 *
 * @param start The starting tensor
 * @param end The ending tensor
 * @param nstep The number of steps with even spacing along the new dimension
 * @param dim Where to insert the new dimension
 * @return Tensor Linearly spaced tensor
 */
[[nodiscard]] Tensor linspace(const Tensor & start, const Tensor & end, Size nstep, Size dim = 0);
/// log-space equivalent of the linspace named constructor
[[nodiscard]] Tensor
logspace(const Tensor & start, const Tensor & end, Size nstep, Size dim = 0, Real base = 10.0);
} // namespace neml2
