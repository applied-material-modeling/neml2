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

#include "neml2/equation_systems/EquationSystem.h"

namespace neml2
{
struct SparseTensorList;

/**
 * @brief Definition of a linear system of equations, Au = b.
 *
 */
class LinearSystem : public EquationSystem
{
public:
  using EquationSystem::EquationSystem;

  void setup() override;

  /// Number of rows in the matrix
  std::size_t m() const;
  /// Number of columns in the matrix
  std::size_t n() const;

  /// Set the unknown u
  virtual void set_u(const SparseTensorList &) = 0;
  /// Get the unknown u
  virtual SparseTensorList u() const = 0;

  /// Assemble and return the operator, A
  SparseTensorList A();
  /// Assemble and return the right-hand side, b
  SparseTensorList b();
  /// Assemble and return the right-hand side and operator
  std::tuple<SparseTensorList, SparseTensorList> A_and_b();

  /// Get the ID-to-unknown mapping for assembly
  const std::vector<LabeledAxisAccessor> & umap() const;
  /// Get the ID-to-unknown-intermediate-shape mapping for assembly
  const std::vector<TensorShape> & intmd_ulayout() const;
  /// Get the ID-to-unknown-base-shape mapping for assembly
  const std::vector<TensorShape> & ulayout() const;

  /// Get the ID-to-RHS mapping for assembly
  const std::vector<LabeledAxisAccessor> & bmap() const;
  /// Get the ID-to-RHS-intermediate-shape mapping for assembly
  const std::vector<TensorShape> & intmd_blayout() const;
  /// Get the ID-to-RHS-base-shape mapping for assembly
  const std::vector<TensorShape> & blayout() const;

protected:
  /// Setup the unknown map
  virtual std::vector<LabeledAxisAccessor> setup_umap() = 0;
  /// Setup the unknown intermediate layout
  virtual std::vector<TensorShape> setup_intmd_ulayout() = 0;
  /// Setup the unknown layout
  virtual std::vector<TensorShape> setup_ulayout() = 0;

  /// Setup the RHS map
  virtual std::vector<LabeledAxisAccessor> setup_bmap() = 0;
  /// Setup the RHS intermediate layout
  virtual std::vector<TensorShape> setup_intmd_blayout() = 0;
  /// Setup the RHS layout
  virtual std::vector<TensorShape> setup_blayout() = 0;

  /**
   * @brief Compute the operator and right-hand side
   *
   * @param A Pointer to the operator matrix -- nullptr if not requested
   * @param b Pointer to the RHS vector -- nullptr if not requested
   */
  virtual void assemble(SparseTensorList * A, SparseTensorList * b) = 0;

  /**
   * @brief Callback before assembly to perform
   *
   * This is useful, for example, to clear obsolete data structures
   *
   * @param A Whether the operator matrix was assembled
   * @param b Whether the RHS vector was assembled
   */
  virtual void pre_assemble(bool A, bool b);

  /**
   * @brief Callback after assembly to perform
   *
   * This is useful, for example, to collect information that isn't available after the first
   * assembly
   *
   * @param A Whether the operator matrix was assembled
   * @param b Whether the RHS vector was assembled
   */
  virtual void post_assemble(bool A, bool b);

  /// Flag indicating if the system matrix is up to date. Setters invalidate this.
  bool _A_up_to_date = false;
  /// Flag indicating if the system RHS is up to date. Setters invalidate this.
  bool _b_up_to_date = false;

  /**
   * @brief The ID-to-unknown mapping
   *
   * The solution vector is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _umap;
  /// ID-to-unknown intermediate shape mapping
  std::optional<std::vector<TensorShape>> _intmd_ulayout;
  /// ID-to-unknown base shape mapping
  std::vector<TensorShape> _ulayout;

  /**
   * @brief The ID-to-RHS mapping
   *
   * The RHS is ordered according to this mapping.
   * This mapping is used by assemble() to collect values in a consistent order.
   */
  std::vector<LabeledAxisAccessor> _bmap;
  /// ID-to-RHS intermediate shape mapping
  std::optional<std::vector<TensorShape>> _intmd_blayout;
  /// ID-to-RHS shape mapping
  std::vector<TensorShape> _blayout;

private:
};

} // namespace neml2
