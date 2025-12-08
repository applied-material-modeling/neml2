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

namespace neml2
{
struct HVector;
struct HMatrix;

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
  LinearSystem & operator=(LinearSystem &&) = default;
  virtual ~LinearSystem() = default;

  ///@{
  /// Set the unknown u from the previous step
  virtual void set_un(const HVector & un) = 0;
  /// Set the given variables g from the current step
  virtual void set_g(const HVector & g) = 0;
  /// Set the given variables g from the previous step
  virtual void set_gn(const HVector & gn) = 0;
  ///@}

  ///@{
  /// Get the unknown u from the previous step
  virtual HVector un() const = 0;
  /// Get the given variables g from the current step
  virtual HVector g() const = 0;
  /// Get the given variables g from the previous step
  virtual HVector gn() const = 0;
  ///@}

  ///@{
  /// Assemble and return the operator, A
  HMatrix A();
  /// Assemble and return the right-hand side, b
  HVector b();
  /// Assemble and return the right-hand side and operator
  std::tuple<HMatrix, HVector> A_and_b();
  ///@}

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
  const std::vector<LabeledAxisAccessor> & bmap() const;
  /// Get the ID-to-RHS-shape mapping for assembly
  const std::vector<TensorShape> & blayout() const;

  /// Create a zero vector for the unknowns
  HVector create_uvec() const;
  /// Create a zero vector for the old solutions
  HVector create_unvec() const;
  /// Create a zero vector for the given variables
  HVector create_gvec() const;
  /// Create a zero vector for the old given variables
  HVector create_gnvec() const;
  /// Create a zero vector for the RHS
  HVector create_bvec() const;

  /// Map a current unknown vector to the old solution vector
  /// This is convenient for making the current unknown vector conformal to the old solution (looking at you pyzag)
  HVector u_to_un(const HVector & u) const;
  /// Map a current given variable vector to the old given variable vector
  /// This is convenient for making the current given variable vector conformal to the old given variable vector (looking at you pyzag)
  HVector g_to_gn(const HVector & g) const;

  /// Map a matrix whose columns are old solutions to a matrix whose columns are current unknowns
  /// This is convenient for making the old Jacobian conformal to the current Jacobian (looking at you pyzag)
  HMatrix un_to_u(const HMatrix & A) const;

protected:
  /// Flag indicating if the system matrix is up to date. Setters invalidate this.
  bool _A_up_to_date = false;
  /// Flag indicating if the system RHS is up to date. Setters invalidate this.
  bool _b_up_to_date = false;

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
  std::vector<LabeledAxisAccessor> _bmap;
  /// ID-to-RHS shape mapping
  std::vector<TensorShape> _blayout;

  /// Map from old solution ID to current unknown ID, -1 indicates no corresponding old solution
  std::vector<int64_t> _un_to_u;
  /// Map from current unknown ID to old solution ID, -1 indicates no corresponding old solution
  std::vector<int64_t> _u_to_un;
  /// Map from old given variable ID to current given variable ID, -1 indicates no corresponding old solution
  std::vector<int64_t> _gn_to_g;
  /// Map from current given variable ID to old given variable ID, -1 indicates no corresponding old given variable
  std::vector<int64_t> _g_to_gn;
};

} // namespace neml2
