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
#include "neml2/tensors/equation_system/Vector.h"
#include "neml2/tensors/equation_system/Matrix.h"

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
  void set_umap(const std::vector<LabeledAxisAccessor> &, const std::vector<TensorShapeRef> &);
  /// Get the ID-to-unknown mapping for assembly
  const std::vector<LabeledAxisAccessor> & umap() const;
  /// Get the ID-to-unknown-shape mapping for assembly
  const std::vector<TensorShape> & ulayout() const;
  /// Set the ID-to-old-solution mapping for assembly
  void set_unmap(const std::vector<LabeledAxisAccessor> &, const std::vector<TensorShapeRef> &);
  /// Get the ID-to-old-solution mapping for assembly
  const std::vector<LabeledAxisAccessor> & unmap() const;
  /// Get the ID-to-old-solution-shape mapping for assembly
  const std::vector<TensorShape> & unlayout() const;

  /// Set the ID-to-prescribed-variable mapping for assembly
  void set_gmap(const std::vector<LabeledAxisAccessor> &, const std::vector<TensorShapeRef> &);
  /// Get the ID-to-prescribed-variable mapping for assembly
  const std::vector<LabeledAxisAccessor> & gmap() const;
  /// Get the ID-to-prescribed-variable-shape mapping for assembly
  const std::vector<TensorShape> & glayout() const;
  /// Set the ID-to-old-prescribed-variable mapping for assembly
  void set_gnmap(const std::vector<LabeledAxisAccessor> &, const std::vector<TensorShapeRef> &);
  /// Get the ID-to-old-prescribed-variable mapping for assembly
  const std::vector<LabeledAxisAccessor> & gnmap() const;
  /// Get the ID-to-old-prescribed-variable-shape mapping for assembly
  const std::vector<TensorShape> & gnlayout() const;

  /// Set the ID-to-residual mapping for assembly
  void set_rmap(const std::vector<LabeledAxisAccessor> &, const std::vector<TensorShapeRef> &);
  /// Get the ID-to-residual mapping for assembly
  const std::vector<LabeledAxisAccessor> & rmap() const;
  /// Get the ID-to-residual-shape mapping for assembly
  const std::vector<TensorShape> & rlayout() const;

  /// Create a zero vector for the unknowns
  es::Vector create_uvec() const;
  /// Create a zero vector for the old solutions
  es::Vector create_unvec() const;
  /// Create a zero vector for the given variables
  es::Vector create_gvec() const;
  /// Create a zero vector for the old given variables
  es::Vector create_gnvec() const;
  /// Create a zero vector for the residuals
  es::Vector create_rvec() const;

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
  /// ID-to-unknown shape mapping
  std::vector<TensorShape> _unknown_shapes;

  /**
   * @brief The ID-to-old-solution mapping
   *
   * The old solution vector is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _old_solutions;
  /// ID-to-old-solution shape mapping
  std::vector<TensorShape> _old_solution_shapes;

  /**
   * @brief The ID-to-given-variable mapping
   *
   * The vector of given variables is ordered according to this mapping.
   */
  std::vector<LabeledAxisAccessor> _given;
  /// ID-to-given shape mapping
  std::vector<TensorShape> _given_shapes;

  /**
   * @brief The ID-to-old-given-variable mapping
   *
   * The vector of old given variables is ordered according to this mapping.
   */
  std::vector<LabeledAxisAccessor> _old_given;
  /// ID-to-old-given shape mapping
  std::vector<TensorShape> _old_given_shapes;

  /**
   * @brief The ID-to-residual mapping
   *
   * The residual is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _residuals;
  /// ID-to-residual shape mapping
  std::vector<TensorShape> _residual_shapes;
};

} // namespace neml2
