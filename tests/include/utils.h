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

#include "neml2/base/Parser.h"
#include "neml2/base/Factory.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/misc/math.h"

/**
 * @brief A simple finite-differencing helper to numerically approximate the derivative of the
 * function at the given point.
 *
 * @tparam F The functor to differentiate
 * @tparam T Type of the input variable, must be _batched_
 * @param f The functor to differentiate, must accept the input of type `Tensor`
 * @param x The point where the derivative is evaluated
 * @param eps The relative perturbation (for each component in the case of non-Scalar input)
 * @param aeps The minimum perturbation to improve numerical stability
 * @return Tensor The derivative at the given point approximated using finite differencing
 */
template <typename F>
[[nodiscard]] neml2::Tensor
finite_differencing_derivative(F && f,
                               const neml2::Tensor & x,
                               neml2::Real eps = 1e-6,
                               neml2::Real aeps = 1e-6)
{
  using namespace neml2;

  // The scalar case is trivial
  if (x.base_dim() == 0)
  {
    auto y0 = Tensor(f(x)).clone();

    auto dx = eps * Scalar(neml2::math::abs(x));
    dx.index_put_({dx < aeps}, aeps);

    auto x1 = x + dx;

    auto y1 = Tensor(f(x1)).clone();
    auto dy_dx = (y1 - y0) / dx;

    return dy_dx;
  }

  // Flatten x to support arbitrarily shaped input
  auto xf = x.base_flatten();
  auto y0 = Tensor(f(x)).clone();
  auto dy_dxf = std::vector<Tensor>(xf.base_size(0));
  for (Size i = 0; i < xf.base_size(0); i++)
  {
    auto dx = eps * Scalar(math::abs(xf.base_index({i})));
    dx.index_put_({dx < aeps}, aeps);

    auto xf1 = xf.clone();
    xf1.base_index_put_({i}, xf1.base_index({i}) + dx);
    auto x1 = Tensor(xf1.reshape(x.sizes()), x.batch_sizes());

    auto y1 = Tensor(f(x1)).clone();
    dy_dxf[i] = (y1 - y0) / dx;
  }

  // Reshape the derivative back to the correct shape
  auto dy_dx = math::base_stack(dy_dxf, -1);
  return dy_dx.base_reshape(utils::add_shapes(y0.base_sizes(), x.base_sizes()));
}
