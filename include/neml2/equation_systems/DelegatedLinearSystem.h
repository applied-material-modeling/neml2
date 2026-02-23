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

#include <optional>

#include "neml2/equation_systems/LinearSystem.h"

namespace neml2
{
/**
 * @brief A lightweight one-group linear system that delegates assembly to pre-supplied blocks.
 *
 * This system is intended for solver composition. It exposes a regular @ref LinearSystem interface
 * so that a child linear solver can be invoked on externally prepared operator/rhs blocks.
 */
class DelegatedLinearSystem : public LinearSystem
{
public:
  struct Layout
  {
    std::vector<LabeledAxisAccessor> map;
    std::vector<TensorShape> intmd;
    std::vector<TensorShape> base;
  };

  DelegatedLinearSystem(const Layout & u_layout,
                        const Layout & g_layout,
                        const Layout & b_layout,
                        const SparseTensorList & A,
                        const std::optional<SparseTensorList> & B = std::nullopt,
                        const std::optional<SparseTensorList> & b = std::nullopt);

  void set_operator(const SparseTensorList & A);
  void set_auxiliary_operator(const std::optional<SparseTensorList> & B);
  void set_rhs(const std::optional<SparseTensorList> & b);

  void set_u(const SparseTensorList &, std::size_t group_idx = 0) override;
  void set_g(const SparseTensorList &) override;

  SparseTensorList u(std::size_t group_idx = 0) const override;
  SparseTensorList g() const override;

protected:
  std::vector<std::vector<LabeledAxisAccessor>> setup_umap() override;
  std::vector<std::vector<TensorShape>> setup_intmd_ulayout() override;
  std::vector<std::vector<TensorShape>> setup_ulayout() override;

  std::vector<LabeledAxisAccessor> setup_gmap() override;
  std::vector<TensorShape> setup_intmd_glayout() override;
  std::vector<TensorShape> setup_glayout() override;

  std::vector<std::vector<LabeledAxisAccessor>> setup_bmap() override;
  std::vector<std::vector<TensorShape>> setup_intmd_blayout() override;
  std::vector<std::vector<TensorShape>> setup_blayout() override;

  void assemble(SparseTensorList * A,
                SparseTensorList * B,
                SparseTensorList * b,
                std::size_t bgroup_idx = 0,
                std::size_t ugroup_idx = 0) override;

private:
  static void validate_group_idx(std::size_t idx, const char * name);
  static void validate_layout(const Layout & layout, const char * name);
  static void validate_size(const SparseTensorList & values, std::size_t expected, const char * name);

  Layout _u_layout;
  Layout _g_layout;
  Layout _b_layout;

  SparseTensorList _A;
  std::optional<SparseTensorList> _B;
  std::optional<SparseTensorList> _b;

  SparseTensorList _u;
  SparseTensorList _g;
};

} // namespace neml2
