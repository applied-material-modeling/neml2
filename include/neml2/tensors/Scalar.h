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

#include "neml2/tensors/PrimitiveTensor.h"

namespace neml2
{
/**
 * @brief Scalar
 *
 * The storage space is (), i.e., scalar.
 *
 */
class Scalar : public PrimitiveTensor<Scalar>
{
public:
  using PrimitiveTensor<Scalar>::PrimitiveTensor;

  Scalar(const CScalar & init, const TensorOptions & options);
};

// The implementation of the single-element accessor depends on the complete definition of Scalar
// However, Scalar is derived from PrimitiveTensor, so we have no choice but to put the
// implementation here.
//
// Practically, whoever wants to use the single-element accessor must also include Scalar.h
template <class Derived, Size... S>
template <typename... Args>
Scalar
PrimitiveTensor<Derived, S...>::operator()(Args... i) const
{
  static_assert(sizeof...(Args) == const_base_dim, "Incorrect number of arguments.");
  static_assert((std::is_convertible_v<Args, Size> && ...),
                "All arguments must be convertible to neml2::Size, aka int64_t");
  return this->base_index({i...});
}
} // namespace neml2
