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

#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/base/OptionSet.h"
#include "neml2/solvers/LinearSystem.h"

namespace neml2
{
/**
 * @brief Definition of a nonlinear system of equations.
 *
 */
class NonlinearSystem
{
public:
  NonlinearSystem(const NonlinearSystem &) = default;
  NonlinearSystem(NonlinearSystem &&) noexcept = default;
  NonlinearSystem & operator=(const NonlinearSystem &) = delete;
  NonlinearSystem & operator=(NonlinearSystem &&) = delete;
  virtual ~NonlinearSystem() = default;

  static OptionSet expected_options();

  NonlinearSystem(const OptionSet & options);

  /// Assemble and return the residual
  es::Vector residual();
  /// Assemble and return the Jacobian
  es::Matrix Jacobian();
  /// Assemble and return the residual and Jacobian
  std::tuple<es::Vector, es::Matrix> residual_and_Jacobian();

  /// Convenient shortcut to set the current solution, assemble and return the residual
  es::Vector residual(const es::Vector & x);
  /// Convenient shortcut to set the current solution, assemble and return the Jacobian
  es::Matrix Jacobian(const es::Vector & x);
  /// Convenient shortcut to set the current guess, assemble and return the residual and Jacobian
  std::tuple<es::Vector, es::Matrix> residual_and_Jacobian(const es::Vector & x);

  /// Set the ID-to-unknown mapping for assembly
  void set_unknown_ordering(const std::vector<LabeledAxisAccessor> & unknowns);
  /// Get the ID-to-unknown mapping for assembly
  const std::vector<LabeledAxisAccessor> & unknown_ordering() const;
  /// Set the ID-to-residual mapping for assembly
  void set_residual_ordering(const std::vector<LabeledAxisAccessor> & residuals);
  /// Get the ID-to-residual mapping for assembly
  const std::vector<LabeledAxisAccessor> & residual_ordering() const;
  /// Set the ID-to-prescribed-variable mapping for assembly
  void set_prescribed_ordering(const std::vector<LabeledAxisAccessor> & prescribeds);
  /// Get the ID-to-prescribed-variable mapping for assembly
  const std::vector<LabeledAxisAccessor> & prescribed_ordering() const;

  /// Set the current solution
  virtual void set_solution(const es::Vector & x) = 0;
  /// Get the current solution
  virtual es::Vector get_solution() const = 0;

protected:
  /**
   * @brief Compute the residual and Jacobian
   *
   * @param r Pointer to the residual vector -- nullptr if not requested
   * @param J Pointer to the Jacobian matrix -- nullptr if not requested
   */
  virtual void assemble(es::Vector * r, es::Matrix * J) = 0;

private:
  /**
   * @brief The ID-to-unknown mapping
   *
   * The solution vector is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _unknowns;

  /**
   * @brief The ID-to-residual mapping
   *
   * The residual is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _residuals;

  /**
   * @brief The ID-to-prescribed-variable mapping
   *
   * The vector of prescribed variables is ordered according to this mapping.
   * This mapping is used by ift() to apply the implicit function theorem.
   */
  std::vector<LabeledAxisAccessor> _prescribeds;
};

} // namespace neml2
