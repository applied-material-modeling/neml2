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

namespace neml2
{
///////////////////////////////////////////////////////////////////////////////
// Addition
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_ADD_SELF(T)                                                                         \
  T operator+(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator+(a, b), utils::broadcast_batch_dim(a, b));                               \
  }                                                                                                \
  static_assert(true)

#define DEFINE_ADD_SYM_SCALAR(T)                                                                   \
  T operator+(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator+(a, b.base_unsqueeze_to(a.base_dim())),                                  \
             utils::broadcast_batch_dim(a, b));                                                    \
  }                                                                                                \
  T operator+(const Scalar & a, const T & b) { return b + a; }                                     \
  static_assert(true)

#define DEFINE_ADD_SYM_REAL(T)                                                                     \
  T operator+(const T & a, const CScalar & b) { return T(at::operator+(a, b), a.batch_sizes()); }  \
  T operator+(const CScalar & a, const T & b) { return b + a; }                                    \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_ADD_SELF);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_ADD_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_ADD_SYM_REAL);
DEFINE_ADD_SELF(Tensor);
DEFINE_ADD_SYM_SCALAR(Tensor);
DEFINE_ADD_SYM_REAL(Tensor);
DEFINE_ADD_SELF(Scalar);
DEFINE_ADD_SYM_REAL(Scalar);

///////////////////////////////////////////////////////////////////////////////
// Subtraction
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_SUB_SELF(T)                                                                         \
  T operator-(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator-(a, b), utils::broadcast_batch_dim(a, b));                               \
  }                                                                                                \
  static_assert(true)

#define DEFINE_SUB_SYM_SCALAR(T)                                                                   \
  T operator-(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator-(a, b.base_unsqueeze_to(a.base_dim())),                                  \
             utils::broadcast_batch_dim(a, b));                                                    \
  }                                                                                                \
  T operator-(const Scalar & a, const T & b)                                                       \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator-(a.base_unsqueeze_to(b.base_dim()), b),                                  \
             utils::broadcast_batch_dim(a, b));                                                    \
  }                                                                                                \
  static_assert(true)

#define DEFINE_SUB_SYM_REAL(T)                                                                     \
  T operator-(const T & a, const CScalar & b) { return T(at::operator-(a, b), a.batch_sizes()); }  \
  T operator-(const CScalar & a, const T & b) { return T(at::operator-(a, b), b.batch_sizes()); }  \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_SUB_SELF);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_SUB_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_SUB_SYM_REAL);
DEFINE_SUB_SELF(Tensor);
DEFINE_SUB_SYM_SCALAR(Tensor);
DEFINE_SUB_SYM_REAL(Tensor);
DEFINE_SUB_SELF(Scalar);
DEFINE_SUB_SYM_REAL(Scalar);

///////////////////////////////////////////////////////////////////////////////
// Multiplication
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_MUL_SELF(T)                                                                         \
  T operator*(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator*(a, b), utils::broadcast_batch_dim(a, b));                               \
  }                                                                                                \
  static_assert(true)

#define DEFINE_MUL_SYM_SCALAR(T)                                                                   \
  T operator*(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator*(a, b.base_unsqueeze_to(a.base_dim())),                                  \
             utils::broadcast_batch_dim(a, b));                                                    \
  }                                                                                                \
  T operator*(const Scalar & a, const T & b) { return b * a; }                                     \
  static_assert(true)

#define DEFINE_MUL_SYM_REAL(T)                                                                     \
  T operator*(const T & a, const CScalar & b) { return T(at::operator*(a, b), a.batch_sizes()); }  \
  T operator*(const CScalar & a, const T & b) { return b * a; }                                    \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_MUL_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_MUL_SYM_REAL);
DEFINE_MUL_SELF(Tensor);
DEFINE_MUL_SYM_SCALAR(Tensor);
DEFINE_MUL_SYM_REAL(Tensor);
DEFINE_MUL_SELF(Scalar);
DEFINE_MUL_SYM_REAL(Scalar);

///////////////////////////////////////////////////////////////////////////////
// Division
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_DIV_SELF(T)                                                                         \
  T operator/(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator/(a, b), utils::broadcast_batch_dim(a, b));                               \
  }                                                                                                \
  static_assert(true)

#define DEFINE_DIV_SYM_SCALAR(T)                                                                   \
  T operator/(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator/(a, b.base_unsqueeze_to(a.base_dim())),                                  \
             utils::broadcast_batch_dim(a, b));                                                    \
  }                                                                                                \
  T operator/(const Scalar & a, const T & b)                                                       \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, b);                                                     \
    return T(at::operator/(a.base_unsqueeze_to(b.base_dim()), b),                                  \
             utils::broadcast_batch_dim(a, b));                                                    \
  }                                                                                                \
  static_assert(true)

#define DEFINE_DIV_SYM_REAL(T)                                                                     \
  T operator/(const T & a, const CScalar & b) { return T(at::operator/(a, b), a.batch_sizes()); }  \
  T operator/(const CScalar & a, const T & b) { return T(at::operator/(a, b), b.batch_sizes()); }  \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_DIV_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_DIV_SYM_REAL);
DEFINE_DIV_SELF(Tensor);
DEFINE_DIV_SYM_SCALAR(Tensor);
DEFINE_DIV_SYM_REAL(Tensor);
DEFINE_DIV_SELF(Scalar);
DEFINE_DIV_SYM_REAL(Scalar);

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
