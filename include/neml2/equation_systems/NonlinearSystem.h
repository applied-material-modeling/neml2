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

#include "neml2/equation_systems/LinearSystem.h"

namespace neml2
{
/**
 * @brief Definition of a nonlinear system of equations, r(u) = 0.
 *
 * Instead of directly defining a nonlinear system, we define the linearized system via its residual
 * and Jacobian, evaluated at a given state u with given variables g, the linearized system is
 * usually expressed as dr(u; g)/du * du = -r(u; g). Rewriting this in the more familiar form Au =
 * b, we have A := dr(u; g)/du and b := -r(u; g).
 *
 * Practically, this means that whenever u or g changes, we need to invalidate the system matrix A
 * and RHS b, and recompute them when requested.
 *
 */
class NonlinearSystem : public LinearSystem
{
public:
  using LinearSystem::LinearSystem;

  void setup() override;

  /// Trigger when unknown variables changed
  virtual void u_changed();
  /// Trigger when given variables changed
  virtual void g_changed();

  /// Set the given variables g from the current step
  virtual void set_g(const SparseTensorList &) = 0;
  /// Get the given variables g from the current step
  virtual SparseTensorList g() const = 0;

  /// Get the ID-to-prescribed-variable mapping for assembly
  const std::vector<LabeledAxisAccessor> & gmap() const;
  /// Get the ID-to-prescribed-variable-intermediate-shape mapping for assembly
  const std::vector<TensorShape> & intmd_glayout() const;
  /// Get the ID-to-prescribed-variable-base-shape mapping for assembly
  const std::vector<TensorShape> & glayout() const;

  /// Assemble the auxiliary matrix B = dr/dg along with A
  virtual std::tuple<SparseTensorList, SparseTensorList> A_and_B();
  /// Assemble the auxiliary matrix B = dr/dg along with A and b
  virtual std::tuple<SparseTensorList, SparseTensorList, SparseTensorList> A_and_B_and_b();
  /// Number of columns in the auxiliary matrix
  std::size_t p() const;

protected:
  /// Setup the given variable map
  virtual std::vector<LabeledAxisAccessor> setup_gmap() = 0;
  /// Setup the given variable intermediate layout
  virtual std::vector<TensorShape> setup_intmd_glayout() = 0;
  /// Setup the given variable base layout
  virtual std::vector<TensorShape> setup_glayout() = 0;

  void post_assemble(bool, bool) override;

  /**
   * @brief The ID-to-given-variable mapping
   *
   * The vector of given variables is ordered according to this mapping.
   */
  std::vector<LabeledAxisAccessor> _gmap;
  /// ID-to-given intermediate shape mapping
  std::optional<std::vector<TensorShape>> _intmd_glayout;
  /// ID-to-given base shape mapping
  std::vector<TensorShape> _glayout;
};

} // namespace neml2
