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

#include <vector>

namespace neml2
{
class Tensor;

/**
 * @brief Use automatic differentiation (AD) to calculate the derivatives of a Tensor w.r.t. another
 * Tensor
 *
 * Torch (and hence NEML2) AD wasn't designed to compute the full Jacobian from the very beginning.
 * Using this method to calculate the full Jacobian is inefficient and is subjected to some
 * restrictions on batch shapes.
 *
 * @param y The `Tensor` to to be differentiated
 * @param xs The arguments to take derivatives with respect to
 * @param retain_graph Whether to retain the computation graph (necessary if y has base storage size
 * > 1)
 * @param create_graph Whether to create the computation graph (necessary if you want to
 * differentiate the returned Jacobian)
 * @param allow_unused Whether to allow unused input argument \p x
 * @return Tensor \f$\partial y/\partial p\f$
 */
std::vector<Tensor> jacrev(const Tensor & y,
                           const std::vector<Tensor> & xs,
                           bool retain_graph = false,
                           bool create_graph = false,
                           bool allow_unused = false);

/// Similar to the other jacrev, but for a single input
Tensor jacrev(const Tensor & y,
              const Tensor & x,
              bool retain_graph = false,
              bool create_graph = false,
              bool allow_unused = false);
} // namespace neml2
