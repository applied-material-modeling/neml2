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

#include "neml2/misc/defaults.h"
#include "neml2/tensors/DTensor.h"
#include "neml2/tensors/functions/einsum.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
///@{
/// Identity map
template <typename T>
DTensor<T, T> imap(const TensorOptions & options = default_tensor_options());
///@}

template <typename T>
DTensor<T, T>
imap(const TensorOptions & /*options*/)
{
  throw NEMLException("Identity map not implemented for this tensor type.");
}

template <>
inline DTensor<Scalar, Scalar>
imap(const TensorOptions & options)
{
  return Scalar::ones(options);
}

template <>
inline DTensor<Vec, Vec>
imap(const TensorOptions & options)
{
  return R2::identity(options);
}

template <>
inline DTensor<R2, R2>
imap(const TensorOptions & options)
{
  return neml2::Tensor::identity(9, options).base_reshape({3, 3, 3, 3});
}

template <>
inline DTensor<SR2, SR2>
imap(const TensorOptions & options)
{
  return SSR4::identity_sym(options);
}

template <>
inline DTensor<SSR4, SSR4>
imap(const TensorOptions & options)
{
  auto I = neml2::Tensor::identity(6, options);
  return einsum("ik,jl", {I, I});
}
} // namespace neml2
