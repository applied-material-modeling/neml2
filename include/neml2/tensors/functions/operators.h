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
#define DECLARE_BINARY_OP(op, T1, T2, TR) TR op(const T1 & a, const T2 & b)
#define DECLARE_BINARY_OP_SELF(op, T) DECLARE_BINARY_OP(op, T, T, T)
#define DECLARE_BINARY_OP_SYM(op, T1, T2, TR)                                                      \
  DECLARE_BINARY_OP(op, T1, T2, TR);                                                               \
  DECLARE_BINARY_OP(op, T2, T1, TR)
#define DECLARE_BINARY_OP_NONCONST(op, T1, T2, TR) TR op(T1 & a, const T2 & b)

///////////////////////////////////////////////////////////////////////////////
// Addition
///////////////////////////////////////////////////////////////////////////////
//       operator+ |  non-scalar-prim tensor scalar real
// ----------------------------------------------------------
// non-scalar-prim |              yes           yes  yes
//          tensor |                     yes    yes  yes
//          scalar |              yes    yes    yes  yes
//            real |              yes    yes    yes
#define DECLARE_ADD_SELF(T) DECLARE_BINARY_OP_SELF(operator+, T)
#define DECLARE_ADD_SYM_SCALAR(T) DECLARE_BINARY_OP_SYM(operator+, T, Scalar, T)
#define DECLARE_ADD_SYM_REAL(T) DECLARE_BINARY_OP_SYM(operator+, T, CScalar, T)
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_ADD_SELF);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_ADD_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_ADD_SYM_REAL);
DECLARE_ADD_SELF(Tensor);
DECLARE_ADD_SYM_SCALAR(Tensor);
DECLARE_ADD_SYM_REAL(Tensor);
DECLARE_ADD_SELF(Scalar);
DECLARE_ADD_SYM_REAL(Scalar);
#undef DECLARE_ADD_SELF
#undef DECLARE_ADD_SYM_SCALAR
#undef DECLARE_ADD_SYM_REAL

///////////////////////////////////////////////////////////////////////////////
// Subtraction
///////////////////////////////////////////////////////////////////////////////
//       operator- |  non-scalar-prim tensor scalar real
// ----------------------------------------------------------
// non-scalar-prim |              yes           yes  yes
//          tensor |                     yes    yes  yes
//          scalar |              yes    yes    yes  yes
//            real |              yes    yes    yes
#define DECLARE_SUB_SELF(T) DECLARE_BINARY_OP_SELF(operator-, T)
#define DECLARE_SUB_SYM_SCALAR(T) DECLARE_BINARY_OP_SYM(operator-, T, Scalar, T)
#define DECLARE_SUB_SYM_REAL(T) DECLARE_BINARY_OP_SYM(operator-, T, CScalar, T)
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_SUB_SELF);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_SUB_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_SUB_SYM_REAL);
DECLARE_SUB_SELF(Tensor);
DECLARE_SUB_SYM_SCALAR(Tensor);
DECLARE_SUB_SYM_REAL(Tensor);
DECLARE_SUB_SELF(Scalar);
DECLARE_SUB_SYM_REAL(Scalar);
#undef DECLARE_SUB_SELF
#undef DECLARE_SUB_SYM_SCALAR
#undef DECLARE_SUB_SYM_REAL

///////////////////////////////////////////////////////////////////////////////
// Multiplication
///////////////////////////////////////////////////////////////////////////////
//       operator* |  non-scalar-prim tensor scalar real
// ----------------------------------------------------------
// non-scalar-prim |                            yes  yes
//          tensor |                     yes    yes  yes
//          scalar |              yes    yes    yes  yes
//            real |              yes    yes    yes
#define DECLARE_MUL_SELF(T) DECLARE_BINARY_OP_SELF(operator*, T)
#define DECLARE_MUL_SYM_SCALAR(T) DECLARE_BINARY_OP_SYM(operator*, T, Scalar, T)
#define DECLARE_MUL_SYM_REAL(T) DECLARE_BINARY_OP_SYM(operator*, T, CScalar, T)
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_MUL_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_MUL_SYM_REAL);
DECLARE_MUL_SELF(Tensor);
DECLARE_MUL_SYM_SCALAR(Tensor);
DECLARE_MUL_SYM_REAL(Tensor);
DECLARE_MUL_SELF(Scalar);
DECLARE_MUL_SYM_REAL(Scalar);
#undef DECLARE_MUL_SELF
#undef DECLARE_MUL_SYM_SCALAR
#undef DECLARE_MUL_SYM_REAL

///////////////////////////////////////////////////////////////////////////////
// Division
///////////////////////////////////////////////////////////////////////////////
//       operator/ |  non-scalar-prim tensor scalar real
// ----------------------------------------------------------
// non-scalar-prim |                            yes  yes
//          tensor |                     yes    yes  yes
//          scalar |              yes    yes    yes  yes
//            real |              yes    yes    yes
#define DECLARE_DIV_SELF(T) DECLARE_BINARY_OP_SELF(operator/, T)
#define DECLARE_DIV_SYM_SCALAR(T) DECLARE_BINARY_OP_SYM(operator/, T, Scalar, T)
#define DECLARE_DIV_SYM_REAL(T) DECLARE_BINARY_OP_SYM(operator/, T, CScalar, T)
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_DIV_SYM_SCALAR);
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_DIV_SYM_REAL);
DECLARE_DIV_SELF(Tensor);
DECLARE_DIV_SYM_SCALAR(Tensor);
DECLARE_DIV_SYM_REAL(Tensor);
DECLARE_DIV_SELF(Scalar);
DECLARE_DIV_SYM_REAL(Scalar);
#undef DECLARE_DIV_SELF
#undef DECLARE_DIV_SYM_SCALAR
#undef DECLARE_DIV_SYM_REAL

///////////////////////////////////////////////////////////////////////////////
// In-place addition
///////////////////////////////////////////////////////////////////////////////
//      operator+= |  non-scalar-prim tensor scalar real
// ----------------------------------------------------------
// non-scalar-prim |                                 yes
//          tensor |                                 yes
//          scalar |                                 yes
//            real |
#define DECLARE_ADD_EQ(T) DECLARE_BINARY_OP_NONCONST(operator+=, T, CScalar, T &)
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_ADD_EQ);
DECLARE_ADD_EQ(Tensor);
DECLARE_ADD_EQ(Scalar);
#undef DECLARE_ADD_EQ

///////////////////////////////////////////////////////////////////////////////
// In-place subtraction
///////////////////////////////////////////////////////////////////////////////
//      operator-= |  non-scalar-prim tensor scalar real
// ----------------------------------------------------------
// non-scalar-prim |                                 yes
//          tensor |                                 yes
//          scalar |                                 yes
//            real |
#define DECLARE_SUB_EQ(T) DECLARE_BINARY_OP_NONCONST(operator-=, T, CScalar, T &)
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_SUB_EQ);
DECLARE_SUB_EQ(Tensor);
DECLARE_SUB_EQ(Scalar);
#undef DECLARE_SUB_EQ

///////////////////////////////////////////////////////////////////////////////
// In-place multiplication
///////////////////////////////////////////////////////////////////////////////
//      operator*= |  non-scalar-prim tensor scalar real
// ----------------------------------------------------------
// non-scalar-prim |                                 yes
//          tensor |                                 yes
//          scalar |                                 yes
//            real |
#define DECLARE_MUL_EQ(T) DECLARE_BINARY_OP_NONCONST(operator*=, T, CScalar, T &)
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_MUL_EQ);
DECLARE_MUL_EQ(Tensor);
DECLARE_MUL_EQ(Scalar);
#undef DECLARE_MUL_EQ

///////////////////////////////////////////////////////////////////////////////
// In-place division
///////////////////////////////////////////////////////////////////////////////
//      operator/= |  non-scalar-prim tensor scalar real
// ----------------------------------------------------------
// non-scalar-prim |                                 yes
//          tensor |                                 yes
//          scalar |                                 yes
//            real |
#define DECLARE_DIV_EQ(T) DECLARE_BINARY_OP_NONCONST(operator/=, T, CScalar, T &)
FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DECLARE_DIV_EQ);
DECLARE_DIV_EQ(Tensor);
DECLARE_DIV_EQ(Scalar);
#undef DECLARE_DIV_EQ

// Undefine macros
#undef DECLARE_BINARY_OP
#undef DECLARE_BINARY_OP_SELF
#undef DECLARE_BINARY_OP_SYM
} // namespace neml2
