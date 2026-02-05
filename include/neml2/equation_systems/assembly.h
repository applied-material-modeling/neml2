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

#include <vector>

#include "neml2/misc/types.h"

namespace neml2
{
class Tensor;
struct SparseTensorList;
template <std::size_t N>
class Derivative;

/// Move the trailing @p dim intermediate dimensions to the base
Tensor pop_intrsc_intmd_dim(const Tensor & t, Size dim);

/// Move the leading @p dim base dimensions to intermediate dimensions
Tensor push_intrsc_intmd_dim(const Tensor & t, Size dim);

/**
 * Move the trailing intrinsic intermediate dimensions to the base for a Derivative If the
 * derivative has broadcasting intrinsic intermediate dimensions, this also handles fullification.
 */
Tensor pop_intrsc_intmd_dim(const Derivative<1> & deriv);

/**
 * Move the trailing intrinsic intermediate dimensions to the base for a Derivative of order @tparam
 * N. This also handles permutation of variable and argument's intrinsic intermediate dimensions and
 * base dimensions.
 */
template <std::size_t N>
Tensor pop_intrsc_intmd_dim(const Tensor & from,
                            const std::array<std::size_t, N> & intrsc_intmd_dims,
                            const std::array<TensorShape, N> & base_shapes,
                            const std::string & debug_name = "<anonymous>");

/**
 * Move the leading base dimensions to intermediate dimensions for a Derivative of order @tparam
 * N. This also handles permutation of variable and argument's intermediate dimensions and base
 * dimensions.
 */
template <std::size_t N>
Tensor push_intrsc_intmd_dim(const Tensor & from,
                             const std::array<std::size_t, N> & intrsc_intmd_dims,
                             const std::array<TensorShape, N> & base_shapes,
                             const std::string & debug_name = "<anonymous>");

/**
 * Convert a tensor from the assembly format to the normal format.
 */
template <std::size_t N>
Tensor from_assembly(const Tensor & from,
                     const std::array<TensorShape, N> & intmd_shapes,
                     const std::array<TensorShape, N> & base_shapes);

/**
 * Convert a tensor from the normal format to the assembly format.
 */
template <std::size_t N>
Tensor to_assembly(const Tensor & from,
                   const std::array<TensorShape, N> & intmd_shapes,
                   const std::array<TensorShape, N> & base_shapes);

/**
 * Disassemble a Tensor with one base dimension into a vector of Tensors according to base shapes
 * and optionally intermediate shapes.
 */
SparseTensorList disassemble(const Tensor &,
                             const std::optional<std::vector<TensorShape>> & intmd_shapes,
                             const std::vector<TensorShape> & base_shapes);

/**
 * Assemble a vector of Tensors into a Tensor with one base dimension according to base shapes
 * and optionally intermediate shapes.
 */
Tensor assemble(const SparseTensorList &,
                const std::optional<std::vector<TensorShape>> & intmd_shapes,
                const std::vector<TensorShape> & base_shapes);

/**
 * Disassemble (in row-major order) a Tensor with two base dimension into a vector of Tensors
 * according to base shapes and optionally intermediate shapes.
 */
SparseTensorList disassemble(const Tensor &,
                             const std::optional<std::vector<TensorShape>> & row_intmd_shapes,
                             const std::optional<std::vector<TensorShape>> & col_intmd_shapes,
                             const std::vector<TensorShape> & row_base_shapes,
                             const std::vector<TensorShape> & col_base_shapes);

/**
 * Assemble a vector of Tensors (in row-major order) into a Tensor with two base dimension according
 * to base shapes and optionally intermediate shapes.
 */
Tensor assemble(const SparseTensorList &,
                const std::optional<std::vector<TensorShape>> & row_intmd_shapes,
                const std::optional<std::vector<TensorShape>> & col_intmd_shapes,
                const std::vector<TensorShape> & row_base_shapes,
                const std::vector<TensorShape> & col_base_shapes);

}
