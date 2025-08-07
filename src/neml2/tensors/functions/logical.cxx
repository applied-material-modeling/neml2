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

#include "neml2/tensors/functions/logical.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/functions/utils.h"

namespace neml2
{
///////////////////////////////////////////////////////////////////////////////
// Logical operators
///////////////////////////////////////////////////////////////////////////////
#define DEFINE_LOGICAL_SELF(T)                                                                     \
  T operator>(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::gt(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                               \
  }                                                                                                \
  T operator<(const T & a, const T & b)                                                            \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::lt(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                               \
  }                                                                                                \
  T operator>=(const T & a, const T & b)                                                           \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::ge(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                               \
  }                                                                                                \
  T operator<=(const T & a, const T & b)                                                           \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::le(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                               \
  }                                                                                                \
  T operator&&(const T & a, const T & b)                                                           \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::logical_and(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                      \
  }                                                                                                \
  T operator||(const T & a, const T & b)                                                           \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::logical_or(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                       \
  }                                                                                                \
  T operator==(const T & a, const T & b)                                                           \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::eq(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                               \
  }                                                                                                \
  T operator!=(const T & a, const T & b)                                                           \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::ne(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                               \
  }                                                                                                \
  T gt(const T & a, const T & b) { return a > b; }                                                 \
  T lt(const T & a, const T & b) { return a < b; }                                                 \
  T ge(const T & a, const T & b) { return a >= b; }                                                \
  T le(const T & a, const T & b) { return a <= b; }                                                \
  T logical_and(const T & a, const T & b) { return a && b; }                                       \
  T logical_or(const T & a, const T & b) { return a || b; }                                        \
  T logical_xor(const T & a, const T & b)                                                          \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::logical_xor(aa, bb), utils::broadcast_dynamic_dim(a, b), i);                      \
  }                                                                                                \
  T eq(const T & a, const T & b) { return a == b; }                                                \
  T ne(const T & a, const T & b) { return a != b; }                                                \
  T operator!(const T & a) { return T(at::logical_not(a), a.dynamic_sizes(), a.intmd_dim()); }     \
  T logical_not(const T & a) { return !a; }                                                        \
  static_assert(true)

#define DEFINE_LOGICAL_SYM_SCALAR(T)                                                               \
  T operator>(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::gt(aa, bb.base_right_unsqueeze_n(a.base_dim())),                                  \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator>(const Scalar & a, const T & b) { return b < a; }                                     \
  T operator<(const T & a, const Scalar & b)                                                       \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::lt(aa, bb.base_right_unsqueeze_n(a.base_dim())),                                  \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator<(const Scalar & a, const T & b) { return b > a; }                                     \
  T operator>=(const T & a, const Scalar & b)                                                      \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::ge(aa, bb.base_right_unsqueeze_n(a.base_dim())),                                  \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator>=(const Scalar & a, const T & b) { return b <= a; }                                   \
  T operator<=(const T & a, const Scalar & b)                                                      \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::le(aa, bb.base_right_unsqueeze_n(a.base_dim())),                                  \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator<=(const Scalar & a, const T & b) { return b >= a; }                                   \
  T operator&&(const T & a, const Scalar & b)                                                      \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::logical_and(aa, bb.base_right_unsqueeze_n(a.base_dim())),                         \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator&&(const Scalar & a, const T & b) { return b && a; }                                   \
  T operator||(const T & a, const Scalar & b)                                                      \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::logical_or(aa, bb.base_right_unsqueeze_n(a.base_dim())),                          \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator||(const Scalar & a, const T & b) { return b || a; }                                   \
  T operator==(const T & a, const Scalar & b)                                                      \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::eq(aa, bb.base_right_unsqueeze_n(a.base_dim())),                                  \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator==(const Scalar & a, const T & b) { return b == a; }                                   \
  T operator!=(const T & a, const Scalar & b)                                                      \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::ne(aa, bb.base_right_unsqueeze_n(a.base_dim())),                                  \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T operator!=(const Scalar & a, const T & b) { return b != a; }                                   \
  T gt(const T & a, const Scalar & b) { return a > b; }                                            \
  T gt(const Scalar & a, const T & b) { return a > b; }                                            \
  T lt(const T & a, const Scalar & b) { return a < b; }                                            \
  T lt(const Scalar & a, const T & b) { return a < b; }                                            \
  T ge(const T & a, const Scalar & b) { return a >= b; }                                           \
  T ge(const Scalar & a, const T & b) { return a >= b; }                                           \
  T le(const T & a, const Scalar & b) { return a <= b; }                                           \
  T le(const Scalar & a, const T & b) { return a <= b; }                                           \
  T logical_and(const T & a, const Scalar & b) { return a && b; }                                  \
  T logical_and(const Scalar & a, const T & b) { return a && b; }                                  \
  T logical_or(const T & a, const Scalar & b) { return a || b; }                                   \
  T logical_or(const Scalar & a, const T & b) { return a || b; }                                   \
  T logical_xor(const T & a, const Scalar & b)                                                     \
  {                                                                                                \
    neml_assert_dynamic_broadcastable_dbg(a, b);                                                   \
    const auto [aa, bb, i] = utils::align_intmd_dim(a, b);                                         \
    return T(at::logical_xor(aa, bb.base_right_unsqueeze_n(a.base_dim())),                         \
             utils::broadcast_dynamic_dim(a, b),                                                   \
             i);                                                                                   \
  }                                                                                                \
  T logical_xor(const Scalar & a, const T & b) { return logical_xor(b, a); }                       \
  T eq(const T & a, const Scalar & b) { return a == b; }                                           \
  T eq(const Scalar & a, const T & b) { return a == b; }                                           \
  T ne(const T & a, const Scalar & b) { return a != b; }                                           \
  T ne(const Scalar & a, const T & b) { return a != b; }                                           \
  static_assert(true)

#define DEFINE_LOGICAL_SYM_CSCALAR(T)                                                              \
  T operator>(const T & a, const CScalar & b)                                                      \
  {                                                                                                \
    return T(at::gt(a, b), a.dynamic_sizes(), a.intmd_dim());                                      \
  }                                                                                                \
  T operator>(const CScalar & a, const T & b) { return b < a; }                                    \
  T operator<(const T & a, const CScalar & b)                                                      \
  {                                                                                                \
    return T(at::lt(a, b), a.dynamic_sizes(), a.intmd_dim());                                      \
  }                                                                                                \
  T operator<(const CScalar & a, const T & b) { return b > a; }                                    \
  T operator>=(const T & a, const CScalar & b)                                                     \
  {                                                                                                \
    return T(at::ge(a, b), a.dynamic_sizes(), a.intmd_dim());                                      \
  }                                                                                                \
  T operator>=(const CScalar & a, const T & b) { return b <= a; }                                  \
  T operator<=(const T & a, const CScalar & b)                                                     \
  {                                                                                                \
    return T(at::le(a, b), a.dynamic_sizes(), a.intmd_dim());                                      \
  }                                                                                                \
  T operator<=(const CScalar & a, const T & b) { return b >= a; }                                  \
  T operator&&(const T & a, const CScalar & b)                                                     \
  {                                                                                                \
    return T(                                                                                      \
        at::logical_and(a, at::scalar_tensor(b, a.options())), a.dynamic_sizes(), a.intmd_dim());  \
  }                                                                                                \
  T operator&&(const CScalar & a, const T & b) { return b && a; }                                  \
  T operator||(const T & a, const CScalar & b)                                                     \
  {                                                                                                \
    return T(                                                                                      \
        at::logical_or(a, at::scalar_tensor(b, a.options())), a.dynamic_sizes(), a.intmd_dim());   \
  }                                                                                                \
  T operator||(const CScalar & a, const T & b) { return b || a; }                                  \
  T operator==(const T & a, const CScalar & b)                                                     \
  {                                                                                                \
    return T(at::eq(a, b), a.dynamic_sizes(), a.intmd_dim());                                      \
  }                                                                                                \
  T operator==(const CScalar & a, const T & b) { return b == a; }                                  \
  T operator!=(const T & a, const CScalar & b)                                                     \
  {                                                                                                \
    return T(at::ne(a, b), a.dynamic_sizes(), a.intmd_dim());                                      \
  }                                                                                                \
  T operator!=(const CScalar & a, const T & b) { return b != a; }                                  \
  T gt(const T & a, const CScalar & b) { return a > b; }                                           \
  T gt(const CScalar & a, const T & b) { return a > b; }                                           \
  T lt(const T & a, const CScalar & b) { return a < b; }                                           \
  T lt(const CScalar & a, const T & b) { return a < b; }                                           \
  T ge(const T & a, const CScalar & b) { return a >= b; }                                          \
  T ge(const CScalar & a, const T & b) { return a >= b; }                                          \
  T le(const T & a, const CScalar & b) { return a <= b; }                                          \
  T le(const CScalar & a, const T & b) { return a <= b; }                                          \
  T logical_and(const T & a, const CScalar & b) { return a && b; }                                 \
  T logical_and(const CScalar & a, const T & b) { return a && b; }                                 \
  T logical_or(const T & a, const CScalar & b) { return a || b; }                                  \
  T logical_or(const CScalar & a, const T & b) { return a || b; }                                  \
  T logical_xor(const T & a, const CScalar & b)                                                    \
  {                                                                                                \
    return T(                                                                                      \
        at::logical_xor(a, at::scalar_tensor(b, a.options())), a.dynamic_sizes(), a.intmd_dim());  \
  }                                                                                                \
  T logical_xor(const CScalar & a, const T & b) { return logical_xor(b, a); }                      \
  T eq(const T & a, const CScalar & b) { return a == b; }                                          \
  T eq(const CScalar & a, const T & b) { return a == b; }                                          \
  T ne(const T & a, const CScalar & b) { return a != b; }                                          \
  T ne(const CScalar & a, const T & b) { return a != b; }                                          \
  static_assert(true)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_LOGICAL_SELF);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_LOGICAL_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEFINE_LOGICAL_SYM_CSCALAR);
DEFINE_LOGICAL_SELF(Tensor);
DEFINE_LOGICAL_SYM_SCALAR(Tensor);
DEFINE_LOGICAL_SYM_CSCALAR(Tensor);
DEFINE_LOGICAL_SELF(Scalar);
DEFINE_LOGICAL_SYM_CSCALAR(Scalar);

} // namespace neml2
