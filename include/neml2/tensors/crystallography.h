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

#include "neml2/tensors/R2.h"
#include "neml2/tensors/Rot.h"

namespace neml2
{
class Scalar;

namespace crystallography
{
/// Helper function to return the symmetry operators given the Orbifold notation
R2 symmetry(const std::string & orbifold, const TensorOptions & options = default_tensor_options());

/// Helper to return all symmetrically-equivalent directions from a cartesian vector
Vec unique_bidirectional(const R2 & ops, const Vec & inp);

/// Calculate the misorientation of two batches of rotations
Scalar misorientation(const Rot & r1, const Rot & r2, const std::string & orbifold = "1");

/// Move a collection of orientations to a fundemental zone defined by the crystal symmetry
// The coice of the reference orientation is arbitrary.  This matches the results from
// Messner, Mark C., and Tianchen Hu. "Fully implicit crystal plasticity models representing
// orientations with modified Rodrigues parameters." Mechanics of Materials (2025): 105388.
//
// This function doesn't tolerate input intermediate dimensions because it needs advanced indexing
//
Rot move_to_fundamental_zone(const Rot & r,
                             const std::string & orbifold,
                             const Rot & ref = Rot::fill(0, 0, 0.005));

namespace symmetry_operators
{
constexpr double a = 0.7071067811865476;
constexpr double b = 0.8660254037844386;
constexpr double h = 0.5;
constexpr double o = 1.0;
constexpr double z = 0.0;

/// @brief tetragonal symmetry operators
Quaternion tetragonal(const TensorOptions & options = default_tensor_options());

/// @brief hexagonal symmetry operators
Quaternion hexagonal(const TensorOptions & options = default_tensor_options());

/// @brief cubic symmetry operators
Quaternion cubic(const TensorOptions & options = default_tensor_options());
} // namespace symmetry_operators
} // namespace crystallography
} // namespace neml2
