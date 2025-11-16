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
template <typename T>
struct imap_t
{
  using type = Tensor;
};

/// Identity map
template <typename T>
DTensor<T, T, typename imap_t<T>::type>
imap(const TensorOptions & options = default_tensor_options());

/// Get the identity map interpreted as the concrete primitive tensor type
template <typename T>
typename imap_t<T>::type imap_v(const TensorOptions & options = default_tensor_options());

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

template <typename T>
DTensor<T, T, typename imap_t<T>::type>
imap(const TensorOptions & /*options*/)
{
  throw NEMLException("Identity map not implemented for this tensor type.");
}

template <typename T>
typename imap_t<T>::type
imap_v(const TensorOptions & options)
{
  return imap<T>(options);
}

// Scalar
template <>
struct imap_t<Scalar>
{
  using type = Scalar;
};
template <>
inline DTensor<Scalar, Scalar, typename imap_t<Scalar>::type>
imap(const TensorOptions & options)
{
  return Scalar::ones(options);
}

// Vec
template <>
struct imap_t<Vec>
{
  using type = R2;
};
template <>
inline DTensor<Vec, Vec, typename imap_t<Vec>::type>
imap(const TensorOptions & options)
{
  return R2::identity(options);
}

// R2
template <>
struct imap_t<R2>
{
  using type = R4;
};
template <>
inline DTensor<R2, R2, typename imap_t<R2>::type>
imap(const TensorOptions & options)
{
  return neml2::Tensor::identity(9, options).base_reshape({3, 3, 3, 3});
}

// SR2
template <>
struct imap_t<SR2>
{
  using type = SSR4;
};
template <>
inline DTensor<SR2, SR2, typename imap_t<SR2>::type>
imap(const TensorOptions & options)
{
  return SSR4::identity_sym(options);
}

// SSR4
template <>
inline DTensor<SSR4, SSR4, typename imap_t<SSR4>::type>
imap(const TensorOptions & options)
{
  auto I = neml2::Tensor::identity(6, options);
  return einsum("ik,jl", {I, I});
}
} // namespace neml2
