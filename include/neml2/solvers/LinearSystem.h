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
#include "neml2/solvers/HVector.h"
#include "neml2/solvers/HMatrix.h"

namespace neml2
{
/**
 * @brief Definition of a linear system of equations, Ax = b.
 *
 */
class LinearSystem
{
public:
  LinearSystem() = default;
  LinearSystem(const LinearSystem &) = default;
  LinearSystem(LinearSystem &&) noexcept = default;
  LinearSystem & operator=(const LinearSystem &) = default;
  LinearSystem & operator=(LinearSystem &&) = default;
  virtual ~LinearSystem() = default;

  /// Assemble and return the operator, A
  HMatrix A();
  /// Assemble and return the right-hand side, b
  HVector b();
  /// Assemble and return the right-hand side and operator
  std::tuple<HMatrix, HVector> A_and_b();

  /// Get the ID-to-unknown mapping for assembly
  const std::vector<LabeledAxisAccessor> & umap() const;
  /// Get the ID-to-unknown-shape mapping for assembly
  const std::vector<TensorShape> & ulayout() const;
  /// Get the ID-to-old-solution mapping for assembly
  const std::vector<LabeledAxisAccessor> & unmap() const;
  /// Get the ID-to-old-solution-shape mapping for assembly
  const std::vector<TensorShape> & unlayout() const;

  /// Get the ID-to-prescribed-variable mapping for assembly
  const std::vector<LabeledAxisAccessor> & gmap() const;
  /// Get the ID-to-prescribed-variable-shape mapping for assembly
  const std::vector<TensorShape> & glayout() const;
  /// Get the ID-to-old-prescribed-variable mapping for assembly
  const std::vector<LabeledAxisAccessor> & gnmap() const;
  /// Get the ID-to-old-prescribed-variable-shape mapping for assembly
  const std::vector<TensorShape> & gnlayout() const;

  /// Get the ID-to-RHS mapping for assembly
  const std::vector<LabeledAxisAccessor> & rmap() const;
  /// Get the ID-to-RHS-shape mapping for assembly
  const std::vector<TensorShape> & rlayout() const;

  /// Create a zero vector for the unknowns
  HVector create_uvec() const;
  /// Create a zero vector for the old solutions
  HVector create_unvec() const;
  /// Create a zero vector for the given variables
  HVector create_gvec() const;
  /// Create a zero vector for the old given variables
  HVector create_gnvec() const;
  /// Create a zero vector for the RHS
  HVector create_rvec() const;

protected:
  /**
   * @brief Compute the operator and right-hand side
   *
   * @param A Pointer to the operator matrix -- nullptr if not requested
   * @param b Pointer to the RHS vector -- nullptr if not requested
   */
  virtual void assemble(HMatrix * A, HVector * b) = 0;

  /**
   * @brief The ID-to-unknown mapping
   *
   * The solution vector is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _umap;
  /// ID-to-unknown shape mapping
  std::vector<TensorShape> _ulayout;

  /**
   * @brief The ID-to-old-solution mapping
   *
   * The old solution vector is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _unmap;
  /// ID-to-old-solution shape mapping
  std::vector<TensorShape> _unlayout;

  /**
   * @brief The ID-to-given-variable mapping
   *
   * The vector of given variables is ordered according to this mapping.
   */
  std::vector<LabeledAxisAccessor> _gmap;
  /// ID-to-given shape mapping
  std::vector<TensorShape> _glayout;

  /**
   * @brief The ID-to-old-given-variable mapping
   *
   * The vector of old given variables is ordered according to this mapping.
   */
  std::vector<LabeledAxisAccessor> _gnmap;
  /// ID-to-old-given shape mapping
  std::vector<TensorShape> _gnlayout;

  /**
   * @brief The ID-to-RHS mapping
   *
   * The RHS is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _rmap;
  /// ID-to-RHS shape mapping
  std::vector<TensorShape> _rlayout;
};

} // namespace neml2
