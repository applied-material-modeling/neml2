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

#include "neml2/solvers/LinearSystem.h"

namespace neml2
{
/**
 * @brief Definition of a nonlinear system of equations, r(x) = 0.
 *
 * Instead of directly defining a nonlinear system, we define the linearized
 * system via its residual and Jacobian, evaluated at a given state x, the
 * linearized system is usually expressed as dr(x)/dx * dx = -r(x). Rewriting
 * this in the more familiar form Ax = b, we have A := dr(x)/dx and b := -r(x).
 *
 */
class NonlinearSystem : public LinearSystem
{
public:
  NonlinearSystem() = default;
  NonlinearSystem(const NonlinearSystem &) = default;
  NonlinearSystem(NonlinearSystem &&) noexcept = default;
  NonlinearSystem & operator=(const NonlinearSystem &) = default;
  NonlinearSystem & operator=(NonlinearSystem &&) = default;
  virtual ~NonlinearSystem() = default;

  using LinearSystem::A;
  /// Convenient shortcut to set the current x, assemble and return the operator
  HMatrix A(const HVector & x);

  using LinearSystem::b;
  /// Convenient shortcut to set the current x, assemble and return the RHS
  HVector b(const HVector & x);

  using LinearSystem::A_and_b;
  /// Convenient shortcut to set the current x, assemble and return the operator and RHS
  std::tuple<HMatrix, HVector> A_and_b(const HVector & x);

  /// Set the current state, x
  virtual void set_x(const HVector & x) = 0;
  /// Get the current state, x
  virtual HVector x() const = 0;
};

} // namespace neml2
