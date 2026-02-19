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
#include "neml2/misc/types.h"

namespace neml2
{
struct SparseTensorList;

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

  virtual void init();

  /// Number of rows in the matrix
  std::size_t m() const;
  /// Number of columns in the matrix
  std::size_t n() const;
  /// Number of columns in the auxiliary matrix
  std::size_t p() const;

  /// Set the unknown u
  virtual void set_u(const SparseTensorList &) = 0;
  /// Set the given variables g from the current step
  virtual void set_g(const SparseTensorList &) = 0;
  /// Get the unknown u
  virtual SparseTensorList u() const = 0;
  /// Get the given variables g from the current step
  virtual SparseTensorList g() const = 0;

  /// Trigger when unknown variables changed
  virtual void u_changed();
  /// Trigger when given variables changed
  virtual void g_changed();

  /// Assemble and return the operator, A
  SparseTensorList A();
  /// Assemble and return the right-hand side, b
  SparseTensorList b();
  /// Assemble and return the right-hand side and operator
  std::tuple<SparseTensorList, SparseTensorList> A_and_b();
  /// Assemble the auxiliary matrix B = dr/dg along with A
  std::tuple<SparseTensorList, SparseTensorList> A_and_B();
  /// Assemble the auxiliary matrix B = dr/dg along with A and b
  std::tuple<SparseTensorList, SparseTensorList, SparseTensorList> A_and_B_and_b();

  /**
   * @brief Get unknown-variable map for assembly.
   *
   * @param group_idx Variable-group index.
   */
  const std::vector<LabeledAxisAccessor> & umap(std::size_t group_idx = 0) const;
  /// Flattened unknown-variable map across all groups.
  std::vector<LabeledAxisAccessor> full_umap() const;
  /// Unknown intermediate layout for the selected group.
  const std::vector<TensorShape> & intmd_ulayout(std::size_t group_idx = 0) const;
  /// Unknown base layout for the selected group.
  const std::vector<TensorShape> & ulayout(std::size_t group_idx = 0) const;

  /**
   * @brief Get given-variable map for assembly.
   */
  const std::vector<LabeledAxisAccessor> & gmap() const;
  /// Given-variable intermediate layout.
  const std::vector<TensorShape> & intmd_glayout() const;
  /// Given-variable base layout.
  const std::vector<TensorShape> & glayout() const;

  /// RHS-variable map for the selected group.
  const std::vector<LabeledAxisAccessor> & bmap(std::size_t group_idx = 0) const;
  /// Flattened RHS-variable map across all groups.
  std::vector<LabeledAxisAccessor> full_bmap() const;
  /// RHS intermediate layout for the selected group.
  const std::vector<TensorShape> & intmd_blayout(std::size_t group_idx = 0) const;
  /// RHS base layout for the selected group.
  const std::vector<TensorShape> & blayout(std::size_t group_idx = 0) const;

  /// Number of unknown-variable groups.
  std::size_t n_ugroup() const;
  /// Number of RHS-variable groups.
  std::size_t n_bgroup() const;

protected:
  /// Setup the unknown map, partitioned by variable group.
  virtual std::vector<std::vector<LabeledAxisAccessor>> setup_umap() = 0;
  /// Setup the unknown intermediate layout, partitioned by variable group.
  virtual std::vector<std::vector<TensorShape>> setup_intmd_ulayout() = 0;
  /// Setup the unknown layout, partitioned by variable group.
  virtual std::vector<std::vector<TensorShape>> setup_ulayout() = 0;

  /// Setup the given variable map.
  virtual std::vector<LabeledAxisAccessor> setup_gmap() = 0;
  /// Setup the given variable intermediate layout.
  virtual std::vector<TensorShape> setup_intmd_glayout() = 0;
  /// Setup the given variable base layout.
  virtual std::vector<TensorShape> setup_glayout() = 0;

  /// Setup the RHS map, partitioned by variable group.
  virtual std::vector<std::vector<LabeledAxisAccessor>> setup_bmap() = 0;
  /// Setup the RHS intermediate layout, partitioned by variable group.
  virtual std::vector<std::vector<TensorShape>> setup_intmd_blayout() = 0;
  /// Setup the RHS layout, partitioned by variable group.
  virtual std::vector<std::vector<TensorShape>> setup_blayout() = 0;

  /**
   * @brief Compute the operator and right-hand side
   *
   * @param A Pointer to the operator matrix -- nullptr if not requested
   * @param B Pointer to the auxiliary matrix -- nullptr if not requested
   * @param b Pointer to the RHS vector -- nullptr if not requested
   */
  virtual void assemble(SparseTensorList * A, SparseTensorList * B, SparseTensorList * b) = 0;

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

  /// ID-to-unknown mapping, partitioned by variable group.
  std::vector<std::vector<LabeledAxisAccessor>> _umap;
  /// ID-to-unknown intermediate shape mapping, partitioned by variable group.
  std::optional<std::vector<std::vector<TensorShape>>> _intmd_ulayout;
  /// ID-to-unknown base shape mapping, partitioned by variable group.
  std::vector<std::vector<TensorShape>> _ulayout;

  /// ID-to-given-variable mapping.
  std::vector<LabeledAxisAccessor> _gmap;
  /// ID-to-given intermediate shape mapping.
  std::optional<std::vector<TensorShape>> _intmd_glayout;
  /// ID-to-given base shape mapping.
  std::vector<TensorShape> _glayout;

  /// ID-to-RHS mapping, partitioned by variable group.
  std::vector<std::vector<LabeledAxisAccessor>> _bmap;
  /// ID-to-RHS intermediate shape mapping, partitioned by variable group.
  std::optional<std::vector<std::vector<TensorShape>>> _intmd_blayout;
  /// ID-to-RHS shape mapping, partitioned by variable group.
  std::vector<std::vector<TensorShape>> _blayout;

private:
  template <typename T>
  static const std::vector<T> & resolve_group(std::size_t group_idx,
                                              const std::vector<std::vector<T>> & group_data,
                                              const std::string & object_name);

  template <typename T>
  static std::vector<T> flatten_groups(const std::vector<std::vector<T>> & group_data);
};

} // namespace neml2
