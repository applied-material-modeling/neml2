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
#include "neml2/tensors/functions/pow.h"

#include <pybind11/operators.h>

#include "python/neml2/csrc/tensors/types.h"

namespace py = pybind11;
using namespace neml2;

void
def_operators(py::module_ & m)
{
#define DEF_BINARY_PYOP(T1, T2, op) c_##T1.def(op(py::self, T2()))
#define DEF_BINARY_PYOP_SYM(T1, T2, op)                                                            \
  c_##T1.def(op(py::self, T2()));                                                                  \
  c_##T1.def(op(T2(), py::self))

// get py types
#define GET_PYTYPE(T) auto c_##T = m.attr(#T).cast<py::class_<T>>()
  FOR_ALL_TENSORBASE(GET_PYTYPE);

/////////////////////////////////////////////////////////////////////////////
//     operator+,- |  non-scalar-prim tensor scalar cscalar
// ----------------------------------------------------------
// non-scalar-prim |              yes           yes  yes
//          tensor |                     yes    yes  yes
//          scalar |              yes    yes    yes  yes
//         cscalar |              yes    yes    yes
//
//     operator*,/ |  non-scalar-prim tensor scalar cscalar
// ----------------------------------------------------------
// non-scalar-prim |                            yes  yes
//          tensor |                     yes    yes  yes
//          scalar |              yes    yes    yes  yes
//         cscalar |              yes    yes    yes
/////////////////////////////////////////////////////////////////////////////
#define DEF_ADD_NONSCALAR_PRIM(T)                                                                  \
  DEF_BINARY_PYOP(T, T, operator+);                                                                \
  DEF_BINARY_PYOP_SYM(T, Scalar, operator+);                                                       \
  DEF_BINARY_PYOP_SYM(T, CScalar, operator+)

  FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEF_ADD_NONSCALAR_PRIM);
  DEF_BINARY_PYOP(Tensor, Tensor, operator+);
  DEF_BINARY_PYOP_SYM(Tensor, Scalar, operator+);
  DEF_BINARY_PYOP_SYM(Tensor, CScalar, operator+);
  DEF_BINARY_PYOP(Scalar, Scalar, operator+);
  DEF_BINARY_PYOP_SYM(Scalar, CScalar, operator+);

#define DEF_SUB_NONSCALAR_PRIM(T)                                                                  \
  DEF_BINARY_PYOP(T, T, operator-);                                                                \
  DEF_BINARY_PYOP_SYM(T, Scalar, operator-);                                                       \
  DEF_BINARY_PYOP_SYM(T, CScalar, operator-)

  FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEF_SUB_NONSCALAR_PRIM);
  DEF_BINARY_PYOP(Tensor, Tensor, operator-);
  DEF_BINARY_PYOP_SYM(Tensor, Scalar, operator-);
  DEF_BINARY_PYOP_SYM(Tensor, CScalar, operator-);
  DEF_BINARY_PYOP(Scalar, Scalar, operator-);
  DEF_BINARY_PYOP_SYM(Scalar, CScalar, operator-);

#define DEF_MUL_NONSCALAR_PRIM(T)                                                                  \
  DEF_BINARY_PYOP_SYM(T, Scalar, operator*);                                                       \
  DEF_BINARY_PYOP_SYM(T, CScalar, operator*)

  FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEF_MUL_NONSCALAR_PRIM);
  DEF_BINARY_PYOP(Tensor, Tensor, operator*);
  DEF_BINARY_PYOP_SYM(Tensor, Scalar, operator*);
  DEF_BINARY_PYOP_SYM(Tensor, CScalar, operator*);
  DEF_BINARY_PYOP(Scalar, Scalar, operator*);
  DEF_BINARY_PYOP_SYM(Scalar, CScalar, operator*);

#define DEF_DIV_NONSCALAR_PRIM(T)                                                                  \
  DEF_BINARY_PYOP_SYM(T, Scalar, operator/);                                                       \
  DEF_BINARY_PYOP_SYM(T, CScalar, operator/)

  FOR_ALL_NONSCALAR_PRIMITIVETENSOR(DEF_DIV_NONSCALAR_PRIM);
  DEF_BINARY_PYOP(Tensor, Tensor, operator/);
  DEF_BINARY_PYOP_SYM(Tensor, Scalar, operator/);
  DEF_BINARY_PYOP_SYM(Tensor, CScalar, operator/);
  DEF_BINARY_PYOP(Scalar, Scalar, operator/);
  DEF_BINARY_PYOP_SYM(Scalar, CScalar, operator/);

/////////////////////////////////////////////////////////////////////////////
//
// operator+=,-=,*=,/= |  non-scalar-prim tensor scalar cscalar
// ----------------------------------------------------------
//     non-scalar-prim |                                 yes
//              tensor |                                 yes
//              scalar |                                 yes
//             cscalar |
/////////////////////////////////////////////////////////////////////////////
#define DEF_INPLACE_TENSORBASE(T)                                                                  \
  DEF_BINARY_PYOP(T, CScalar, operator+=);                                                         \
  DEF_BINARY_PYOP(T, CScalar, operator-=);                                                         \
  DEF_BINARY_PYOP(T, CScalar, operator*=);                                                         \
  DEF_BINARY_PYOP(T, CScalar, operator/=)
  FOR_ALL_TENSORBASE(DEF_INPLACE_TENSORBASE);

/////////////////////////////////////////////////////////////////////////////
// operator- (unary)
/////////////////////////////////////////////////////////////////////////////
#define DEF_UNARY_NEG_TENSORBASE(T) c_##T.def(-py::self)
  FOR_ALL_TENSORBASE(DEF_UNARY_NEG_TENSORBASE);

/////////////////////////////////////////////////////////////////////////////
// pow
/////////////////////////////////////////////////////////////////////////////
#define DEF_POW_TENSORBASE(T)                                                                      \
  c_##T.def(                                                                                       \
      "__pow__",                                                                                   \
      [](const T & a, const CScalar & n, const py::object & mod)                                   \
      {                                                                                            \
        if (!mod.is_none())                                                                        \
          throw py::type_error("mod argument not supported");                                      \
        return neml2::pow(a, n);                                                                   \
      },                                                                                           \
      py::is_operator())
  FOR_ALL_TENSORBASE(DEF_POW_TENSORBASE);

#define DEF_RPOW(T)                                                                                \
  c_##T.def(                                                                                       \
      "__rpow__",                                                                                  \
      [](const T & n, const CScalar & a, const py::object & mod)                                   \
      {                                                                                            \
        if (!mod.is_none())                                                                        \
          throw py::type_error("mod argument not supported");                                      \
        return neml2::pow(a, n);                                                                   \
      },                                                                                           \
      py::is_operator())
  DEF_RPOW(Scalar);
  DEF_RPOW(Tensor);

  // element-wise pow
  c_Tensor.def(
      "__pow__",
      [](const Tensor & a, const Tensor & n, const py::object & mod)
      {
        if (!mod.is_none())
          throw py::type_error("mod argument not supported");
        return neml2::pow(a, n);
      },
      py::is_operator());
}
