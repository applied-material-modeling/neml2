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
#include "neml2/tensors/tensors_fwd.h"

namespace neml2
{
/**
 * @brief Create a new tensor by adding a new dimension with logarithmic spacing between \p
 * start and \p end.
 *
 * \p start and \p end must be broadcastable. The new dimension will be added at the
 * user-specified dimension \p dim which defaults to 0.
 *
 * For example, if \p start has shape `(3, 2; 5, 5)` and \p end has shape `(3, 1; 5, 5)`, then
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cpp
 * logspace(start, end, 100, 1);
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * will have shape `(3, 100, 2; 5, 5)`, note the location of the new dimension and the
 * broadcasting.
 *
 * @param start The starting tensor
 * @param end The ending tensor
 * @param nstep The number of steps with even spacing along the new dimension
 * @param dim Where to insert the new dimension
 * @param base The base of the logarithm
 * @return Tensor Logarithmically spaced tensor
 */
///@{
#define DECLARE_LOGSPACE(T)                                                                        \
  T dynamic_logspace(                                                                              \
      const T & start, const T & end, Size nstep, Size dim = 0, const CScalar & base = 10.0);      \
  T intmd_logspace(                                                                                \
      const T & start, const T & end, Size nstep, Size dim = 0, const CScalar & base = 10.0)
FOR_ALL_TENSORBASE(DECLARE_LOGSPACE);
#undef DECLARE_LOGSPACE

Tensor base_logspace(const Tensor & start,
                     const Tensor & end,
                     Size nstep,
                     Size dim = 0,
                     const CScalar & base = 10.0);
///@}
} // namespace neml2
