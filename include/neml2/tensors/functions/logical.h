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

#include <ATen/TensorOperators.h>

#include "neml2/misc/types.h"
#include "neml2/tensors/macros.h"
#include "neml2/tensors/tensors_fwd.h"

namespace neml2
{
// Forward declaration
#define FORWARD_DECLARATION(T) class T
FOR_ALL_TENSORBASE(FORWARD_DECLARATION);

// Define macros (let's be responsible and undefine them afterwards)
#define DECLARE_UNARY_OP(op, T, TR) TR op(const T & a)
#define DECLARE_BINARY_OP(op, T1, T2, TR) TR op(const T1 & a, const T2 & b)
#define DECLARE_BINARY_OP_SELF(op, T) DECLARE_BINARY_OP(op, T, T, T)
#define DECLARE_BINARY_OP_SYM(op, T1, T2, TR)                                                      \
  DECLARE_BINARY_OP(op, T1, T2, TR);                                                               \
  DECLARE_BINARY_OP(op, T2, T1, TR)

///////////////////////////////////////////////////////////////////////////////
// Logical operators
///////////////////////////////////////////////////////////////////////////////
//                 |  non-scalar-prim tensor scalar cscalar
// ----------------------------------------------------------
// non-scalar-prim |              yes           yes  yes
//          tensor |                     yes    yes  yes
//          scalar |              yes    yes    yes  yes
//         cscalar |              yes    yes    yes
#define DECLARE_LOGICAL_SELF(T)                                                                    \
  DECLARE_BINARY_OP_SELF(operator>, T);                                                            \
  DECLARE_BINARY_OP_SELF(operator<, T);                                                            \
  DECLARE_BINARY_OP_SELF(operator>=, T);                                                           \
  DECLARE_BINARY_OP_SELF(operator<=, T);                                                           \
  DECLARE_BINARY_OP_SELF(operator&&, T);                                                           \
  DECLARE_BINARY_OP_SELF(operator||, T);                                                           \
  DECLARE_BINARY_OP_SELF(operator==, T);                                                           \
  DECLARE_BINARY_OP_SELF(operator!=, T);                                                           \
                                                                                                   \
  DECLARE_BINARY_OP_SELF(gt, T);                                                                   \
  DECLARE_BINARY_OP_SELF(lt, T);                                                                   \
  DECLARE_BINARY_OP_SELF(ge, T);                                                                   \
  DECLARE_BINARY_OP_SELF(le, T);                                                                   \
  DECLARE_BINARY_OP_SELF(logical_and, T);                                                          \
  DECLARE_BINARY_OP_SELF(logical_or, T);                                                           \
  DECLARE_BINARY_OP_SELF(logical_xor, T);                                                          \
  DECLARE_BINARY_OP_SELF(eq, T);                                                                   \
  DECLARE_BINARY_OP_SELF(ne, T);                                                                   \
                                                                                                   \
  DECLARE_UNARY_OP(operator!, T, T);                                                               \
  DECLARE_UNARY_OP(logical_not, T, T)

#define DECLARE_LOGICAL_SCALAR(T)                                                                  \
  DECLARE_BINARY_OP_SYM(operator>, T, Scalar, T);                                                  \
  DECLARE_BINARY_OP_SYM(operator<, T, Scalar, T);                                                  \
  DECLARE_BINARY_OP_SYM(operator>=, T, Scalar, T);                                                 \
  DECLARE_BINARY_OP_SYM(operator<=, T, Scalar, T);                                                 \
  DECLARE_BINARY_OP_SYM(operator&&, T, Scalar, T);                                                 \
  DECLARE_BINARY_OP_SYM(operator||, T, Scalar, T);                                                 \
  DECLARE_BINARY_OP_SYM(operator==, T, Scalar, T);                                                 \
  DECLARE_BINARY_OP_SYM(operator!=, T, Scalar, T);                                                 \
                                                                                                   \
  DECLARE_BINARY_OP_SYM(gt, T, Scalar, T);                                                         \
  DECLARE_BINARY_OP_SYM(lt, T, Scalar, T);                                                         \
  DECLARE_BINARY_OP_SYM(ge, T, Scalar, T);                                                         \
  DECLARE_BINARY_OP_SYM(le, T, Scalar, T);                                                         \
  DECLARE_BINARY_OP_SYM(logical_and, T, Scalar, T);                                                \
  DECLARE_BINARY_OP_SYM(logical_or, T, Scalar, T);                                                 \
  DECLARE_BINARY_OP_SYM(logical_xor, T, Scalar, T);                                                \
  DECLARE_BINARY_OP_SYM(eq, T, Scalar, T);                                                         \
  DECLARE_BINARY_OP_SYM(ne, T, Scalar, T)

#define DECLARE_LOGICAL_CSCALAR(T)                                                                 \
  DECLARE_BINARY_OP_SYM(operator>, T, CScalar, T);                                                 \
  DECLARE_BINARY_OP_SYM(operator<, T, CScalar, T);                                                 \
  DECLARE_BINARY_OP_SYM(operator>=, T, CScalar, T);                                                \
  DECLARE_BINARY_OP_SYM(operator<=, T, CScalar, T);                                                \
  DECLARE_BINARY_OP_SYM(operator&&, T, CScalar, T);                                                \
  DECLARE_BINARY_OP_SYM(operator||, T, CScalar, T);                                                \
  DECLARE_BINARY_OP_SYM(operator==, T, CScalar, T);                                                \
  DECLARE_BINARY_OP_SYM(operator!=, T, CScalar, T);                                                \
                                                                                                   \
  DECLARE_BINARY_OP_SYM(gt, T, CScalar, T);                                                        \
  DECLARE_BINARY_OP_SYM(lt, T, CScalar, T);                                                        \
  DECLARE_BINARY_OP_SYM(ge, T, CScalar, T);                                                        \
  DECLARE_BINARY_OP_SYM(le, T, CScalar, T);                                                        \
  DECLARE_BINARY_OP_SYM(logical_and, T, CScalar, T);                                               \
  DECLARE_BINARY_OP_SYM(logical_or, T, CScalar, T);                                                \
  DECLARE_BINARY_OP_SYM(logical_xor, T, CScalar, T);                                               \
  DECLARE_BINARY_OP_SYM(eq, T, CScalar, T);                                                        \
  DECLARE_BINARY_OP_SYM(ne, T, CScalar, T)

FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_LOGICAL_SELF);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_LOGICAL_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_LOGICAL_CSCALAR);
DECLARE_LOGICAL_SELF(Tensor);
DECLARE_LOGICAL_SCALAR(Tensor);
DECLARE_LOGICAL_CSCALAR(Tensor);
DECLARE_LOGICAL_SELF(Scalar);
DECLARE_LOGICAL_CSCALAR(Scalar);

#undef DECLARE_LOGICAL_SELF
#undef DECLARE_LOGICAL_SCALAR
#undef DECLARE_LOGICAL_CSCALAR

// Undefine macros
#undef DECLARE_BINARY_OP
#undef DECLARE_BINARY_OP_SELF
#undef DECLARE_BINARY_OP_SYM
} // namespace neml2
