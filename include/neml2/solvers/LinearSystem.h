// Copyright 2024, UChicago Argonne, LLC
// All Rights Rerved
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
#include "neml2/tensors/Tensor.h"

namespace neml2::es
{
struct Vector : public std::vector<Tensor>
{
  Vector() = default;

  /// Conversion from vector of Tensors must be explicit
  explicit Vector(std::vector<Tensor> r)
    : std::vector<Tensor>(std::move(r))
  {
  }

  /// Whether any of the contained Tensors require gradients
  bool requires_grad() const;

  /// Negation
  Vector operator-() const;

  /// Update the data of the contained Tensors (without changing their computational graph)
  void update_data(const Vector & other);

  /// Update the contained Tensors
  void update(const Vector & other);
};

struct Matrix : public std::vector<std::vector<Tensor>>
{
  Matrix() = default;

  /// Conversion from matrix of Tensors must be explicit
  explicit Matrix(std::vector<std::vector<Tensor>> J)
    : std::vector<std::vector<Tensor>>(std::move(J))
  {
  }
};

/// squared vector-norm of an es::Vector
Scalar norm_sq(const Vector & v);

/// vector-norm of an es::Vector
Scalar norm(const Vector & v);
} // namespace neml2::es
