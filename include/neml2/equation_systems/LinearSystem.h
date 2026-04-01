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

#include "neml2/equation_systems/AxisLayout.h"

namespace neml2
{
struct AssembledVector;
struct AssembledMatrix;

/**
 * @brief Definition of a linear system of equations, Au = b.
 *
 */
class LinearSystem
{
public:
  LinearSystem() = default;
  LinearSystem(const LinearSystem &) = default;
  LinearSystem(LinearSystem &&) noexcept = default;
  LinearSystem & operator=(const LinearSystem &) = default;
  LinearSystem & operator=(LinearSystem &&) noexcept = default;
  virtual ~LinearSystem() = default;

  /// Setup axis layouts
  virtual void init();

  /// Set the unknown u
  virtual void set_u(const AssembledVector &) = 0;
  /// Set the given variables g from the current step
  virtual void set_g(const AssembledVector &) = 0;
  /// Get the unknown u
  virtual AssembledVector u() const = 0;
  /// Get the given variables g from the current step
  virtual AssembledVector g() const = 0;

  /// Trigger when unknown variables changed
  virtual void u_changed();
  /// Trigger when given variables changed
  virtual void g_changed();

  /// Assemble and return the operator, A
  AssembledMatrix A();
  /// Assemble and return the right-hand side, b
  AssembledVector b();
  /// Assemble and return the right-hand side and operator
  std::tuple<AssembledMatrix, AssembledVector> A_and_b();
  /// Assemble the auxiliary matrix B = dr/dg along with A
  std::tuple<AssembledMatrix, AssembledMatrix> A_and_B();
  /// Assemble the auxiliary matrix B = dr/dg along with A and b
  std::tuple<AssembledMatrix, AssembledMatrix, AssembledVector> A_and_B_and_b();

  /// Get the unknown-variable layout
  AxisLayout ulayout() const;
  /// Get the given-variable layout
  AxisLayout glayout() const;
  /// Get the RHS variable layout
  AxisLayout blayout() const;

protected:
  /// Setup the unknown layout, partitioned by variable group.
  virtual std::shared_ptr<AxisLayout> setup_ulayout() = 0;
  /// Setup the given variable layout
  virtual std::shared_ptr<AxisLayout> setup_glayout() = 0;
  /// Setup the RHS variable layout
  virtual std::shared_ptr<AxisLayout> setup_blayout() = 0;

  /**
   * @brief Compute the operator and right-hand side
   *
   * @param A Pointer to the operator matrix -- nullptr if not requested
   * @param B Pointer to the auxiliary matrix -- nullptr if not requested
   * @param b Pointer to the RHS vector -- nullptr if not requested
   */
  virtual void assemble(AssembledMatrix * A, AssembledMatrix * B, AssembledVector * b) = 0;

  /**
   * @brief Callback before assembly to perform
   *
   * This is useful, for example, to clear obsolete data structures
   *
   * @param A Whether the operator matrix was assembled
   * @param B Whether the auxiliary matrix was assembled
   * @param b Whether the RHS vector was assembled
   */
  virtual void pre_assemble(bool A, bool B, bool b);

  /**
   * @brief Callback after assembly to perform
   *
   * This is useful, for example, to collect information that isn't available after the first
   * assembly
   *
   * @param A Whether the operator matrix was assembled
   * @param B Whether the auxiliary matrix was assembled
   * @param b Whether the RHS vector was assembled
   */
  virtual void post_assemble(bool A, bool B, bool b);

  /// Flag indicating if the system matrix is up to date. Setters invalidate this.
  bool _A_up_to_date = false;
  /// Flag indicating if the auxiliary matrix is up to date. Setters invalidate this.
  bool _B_up_to_date = false;
  /// Flag indicating if the system RHS is up to date. Setters invalidate this.
  bool _b_up_to_date = false;

  /// Layout of unknowns, partitioned by variable groups
  std::shared_ptr<AxisLayout> _ulayout;
  /// Layout of given variables
  std::shared_ptr<AxisLayout> _glayout;
  /// Layout of RHS variables, partitioned by variable groups
  std::shared_ptr<AxisLayout> _blayout;
};

} // namespace neml2
