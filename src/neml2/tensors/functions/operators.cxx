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

#include "neml2/tensors/functions/operators.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/functions/utils.h"

namespace neml2
{
///////////////////////////////////////////////////////////////////////////////
// Addition
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_ADD_SELF(T)                                                                         \
  T operator+(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator+(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                        \
  }                                                                                                \
  static_assert(true)

#define DEFINE_ADD_SYM_SCALAR(T)                                                                   \
  T operator+(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator+(aa, bb.base_unsqueeze_n(a.base_dim(), -1)),                             \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator+(const Scalar & a, const T & b) { return b + a; }                                     \
  static_assert(true)

#define DEFINE_ADD_SYM_CSCALAR(T)                                                                  \
  T operator+(const T & a, const CScalar & b)                                                      \
  {                                                                                                \
    return T(at::operator+(a, b), a.dynamic_sizes(), a.intmd_dim());                               \
  }                                                                                                \
  T operator+(const CScalar & a, const T & b) { return b + a; }                                    \
  static_assert(true)

Tensor
operator+(const Tensor & a, const Tensor & b)
{
  neml_assert_dynamic_broadcastable_dbg(a, b);
  const auto [aa, bb, i] = utils::align_static_dim(a, b);
  return Tensor(at::operator+(aa, bb), utils::broadcast_dynamic_dim(a, b), i);
}

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_ADD_SELF);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_ADD_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_ADD_SYM_CSCALAR);
DEFINE_ADD_SYM_SCALAR(Tensor);
DEFINE_ADD_SYM_CSCALAR(Tensor);
DEFINE_ADD_SELF(Scalar);
DEFINE_ADD_SYM_CSCALAR(Scalar);

///////////////////////////////////////////////////////////////////////////////
// Subtraction
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_SUB_SELF(T)                                                                         \
  T operator-(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator-(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                        \
  }                                                                                                \
  static_assert(true)

#define DEFINE_SUB_SYM_SCALAR(T)                                                                   \
  T operator-(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator-(aa, bb.base_unsqueeze_n(a.base_dim(), -1)),                             \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator-(const Scalar & a, const T & b)                                                       \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator-(aa.base_unsqueeze_n(b.base_dim(), -1), bb),                             \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  static_assert(true)

#define DEFINE_SUB_SYM_CSCALAR(T)                                                                  \
  T operator-(const T & a, const CScalar & b)                                                      \
  {                                                                                                \
    return T(at::operator-(a, b), a.dynamic_sizes(), a.intmd_dim());                               \
  }                                                                                                \
  T operator-(const CScalar & a, const T & b)                                                      \
  {                                                                                                \
    return T(at::operator-(a, b), b.dynamic_sizes(), b.intmd_dim());                               \
  }                                                                                                \
  static_assert(true)

Tensor
operator-(const Tensor & a, const Tensor & b)
{
  neml_assert_dynamic_broadcastable_dbg(a, b);
  const auto [aa, bb, i] = utils::align_static_dim(a, b);
  return Tensor(at::operator-(aa, bb), utils::broadcast_dynamic_dim(a, b), i);
}

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_SUB_SELF);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_SUB_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_SUB_SYM_CSCALAR);
DEFINE_SUB_SYM_SCALAR(Tensor);
DEFINE_SUB_SYM_CSCALAR(Tensor);
DEFINE_SUB_SELF(Scalar);
DEFINE_SUB_SYM_CSCALAR(Scalar);

///////////////////////////////////////////////////////////////////////////////
// Multiplication
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_MUL_SELF(T)                                                                         \
  T operator*(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator*(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                        \
  }                                                                                                \
  static_assert(true)

#define DEFINE_MUL_SYM_SCALAR(T)                                                                   \
  T operator*(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator*(aa, bb.base_unsqueeze_n(a.base_dim(), -1)),                             \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator*(const Scalar & a, const T & b) { return b * a; }                                     \
  static_assert(true)

#define DEFINE_MUL_SYM_CSCALAR(T)                                                                  \
  T operator*(const T & a, const CScalar & b)                                                      \
  {                                                                                                \
    return T(at::operator*(a, b), a.dynamic_sizes(), a.intmd_dim());                               \
  }                                                                                                \
  T operator*(const CScalar & a, const T & b) { return b * a; }                                    \
  static_assert(true)

Tensor
operator*(const Tensor & a, const Tensor & b)
{
  neml_assert_dynamic_broadcastable_dbg(a, b);
  const auto [aa, bb, i] = utils::align_static_dim(a, b);
  return Tensor(at::operator*(aa, bb), utils::broadcast_dynamic_dim(a, b), i);
}

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_MUL_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_MUL_SYM_CSCALAR);
DEFINE_MUL_SYM_SCALAR(Tensor);
DEFINE_MUL_SYM_CSCALAR(Tensor);
DEFINE_MUL_SELF(Scalar);
DEFINE_MUL_SYM_CSCALAR(Scalar);

///////////////////////////////////////////////////////////////////////////////
// Division
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_DIV_SELF(T)                                                                         \
  T operator/(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator/(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                        \
  }                                                                                                \
  static_assert(true)

#define DEFINE_DIV_SYM_SCALAR(T)                                                                   \
  T operator/(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator/(aa, bb.base_unsqueeze_n(a.base_dim(), -1)),                             \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator/(const Scalar & a, const T & b)                                                       \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::operator/(aa.base_unsqueeze_n(b.base_dim(), -1), b),                              \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  static_assert(true)

#define DEFINE_DIV_SYM_CSCALAR(T)                                                                  \
  T operator/(const T & a, const CScalar & b)                                                      \
  {                                                                                                \
    return T(at::operator/(a, b), a.dynamic_sizes(), a.intmd_dim());                               \
  }                                                                                                \
  T operator/(const CScalar & a, const T & b)                                                      \
  {                                                                                                \
    return T(at::operator/(a, b), b.dynamic_sizes(), b.intmd_dim());                               \
  }                                                                                                \
  static_assert(true)

Tensor
operator/(const Tensor & a, const Tensor & b)
{
  neml_assert_dynamic_broadcastable_dbg(a, b);
  const auto [aa, bb, i] = utils::align_static_dim(a, b);
  return Tensor(at::operator/(aa, bb), utils::broadcast_dynamic_dim(a, b), i);
}

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_DIV_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_DIV_SYM_CSCALAR);
DEFINE_DIV_SYM_SCALAR(Tensor);
DEFINE_DIV_SYM_CSCALAR(Tensor);
DEFINE_DIV_SELF(Scalar);
DEFINE_DIV_SYM_CSCALAR(Scalar);

///////////////////////////////////////////////////////////////////////////////
// In-place addition
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_ADD_EQ(T)                                                                           \
  T & operator+=(T & a, const CScalar & b)                                                         \
  {                                                                                                \
    at::Tensor(a) += b;                                                                            \
    return a;                                                                                      \
  }                                                                                                \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_ADD_EQ);
DEFINE_ADD_EQ(Tensor);
DEFINE_ADD_EQ(Scalar);

///////////////////////////////////////////////////////////////////////////////
// In-place subtraction
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_SUB_EQ(T)                                                                           \
  T & operator-=(T & a, const CScalar & b)                                                         \
  {                                                                                                \
    at::Tensor(a) -= b;                                                                            \
    return a;                                                                                      \
  }                                                                                                \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_SUB_EQ);
DEFINE_SUB_EQ(Tensor);
DEFINE_SUB_EQ(Scalar);

///////////////////////////////////////////////////////////////////////////////
// In-place multiplication
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_MUL_EQ(T)                                                                           \
  T & operator*=(T & a, const CScalar & b)                                                         \
  {                                                                                                \
    at::Tensor(a) *= b;                                                                            \
    return a;                                                                                      \
  }                                                                                                \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_MUL_EQ);
DEFINE_MUL_EQ(Tensor);
DEFINE_MUL_EQ(Scalar);

///////////////////////////////////////////////////////////////////////////////
// In-place division
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_DIV_EQ(T)                                                                           \
  T & operator/=(T & a, const CScalar & b)                                                         \
  {                                                                                                \
    at::Tensor(a) /= b;                                                                            \
    return a;                                                                                      \
  }                                                                                                \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_DIV_EQ);
DEFINE_DIV_EQ(Tensor);
DEFINE_DIV_EQ(Scalar);
} // namespace neml2
