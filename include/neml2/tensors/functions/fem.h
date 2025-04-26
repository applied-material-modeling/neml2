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

#include "neml2/tensors/Tensor.h"

namespace neml2
{
/**
 * @brief Scatter a vector associated with all degrees of freedom to a tensor specified by a DOF
 * map.
 *
 * \p dof_map must be of integer type, and its values represent the DOF indices to which the values
 * of \p v should be scattered.
 *
 * The input vector \p v should be a "vector" of shape (; Ndof) where Ndof is the number of
 * degrees of freedom. The shape of \p dof_map is (Nelem; Ndofe, Nvar) representing
 * the scattering pattern, where Nelem is the number of elements in the mesh, Ndofe is the number of
 * DOF objects (i.e., nodes for Lagrange polynomial basis) per element, and Nvar is the number of
 * variables/functions associated with each DOF object.
 *
 * @returns a tensor with shape (Nelem; Ndofe, Nvar).
 */
Tensor fem_scatter(const Tensor & v, const Tensor & dof_map);

/**
 * @brief Interpolate a tensor of values given evaluated basis functions.
 *
 * The input tensor \p elem_dofs must be of shape (Nelem; Ndofe, Nvar) where Nelem is the
 * number of elements in the mesh, Ndofe is the number of DOF objects (i.e., nodes) per element, and
 * Nvar is the number of variables/functions to be interpolated. The \p basis should have zero or
 * one batch dimension and at least one base dimension, i.e. of shape (?; Ndofe, ...), and should be
 * batch-broadcastable to \p elem_dofs. The first base dimension should have size Ndofe. The
 * remaining base shape (...) represents the points of interpolation.
 *
 * @returns a tensor with shape (Nelem; Nvar, ...).
 */
Tensor fem_interpolate(const Tensor & elem_dofs, const Tensor & basis);

/**
 * @brief This is the inverse operation of fem_scatter. It assembles a scattered vector into a
 * tensor given a DOF map.
 *
 * @returns a tensor with shape (; Ndof)
 */
Tensor fem_assemble(const ATensor & v_scattered, const ATensor & dof_map, Size ndof);
} // namespace neml2
