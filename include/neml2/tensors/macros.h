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

/// @name Macros for all tensor types with semicolon as the separator
///@{
#define FOR_ALL_VECBASE(f)                                                                         \
  f(Vec);                                                                                          \
  f(Rot);                                                                                          \
  f(WR2)

#define FOR_ALL_R2BASE(f) f(R2)

#define FOR_ALL_NONSCALAR_PRIMITIVETENSOR(f)                                                       \
  FOR_ALL_VECBASE(f);                                                                              \
  FOR_ALL_R2BASE(f);                                                                               \
  f(SR2);                                                                                          \
  f(R3);                                                                                           \
  f(SFR3);                                                                                         \
  f(R4);                                                                                           \
  f(SFR4);                                                                                         \
  f(WFR4);                                                                                         \
  f(SSR4);                                                                                         \
  f(SFFR4);                                                                                        \
  f(SWR4);                                                                                         \
  f(WSR4);                                                                                         \
  f(WWR4);                                                                                         \
  f(R5);                                                                                           \
  f(SSFR5);                                                                                        \
  f(Quaternion);                                                                                   \
  f(MillerIndex);                                                                                  \
  f(SSSSR8);                                                                                       \
  f(R8)

#define FOR_ALL_NONSCALAR_TENSORBASE(f)                                                            \
  FOR_ALL_NONSCALAR_PRIMITIVETENSOR(f);                                                            \
  f(Tensor)

#define FOR_ALL_PRIMITIVETENSOR(f)                                                                 \
  FOR_ALL_NONSCALAR_PRIMITIVETENSOR(f);                                                            \
  f(Scalar)

#define FOR_ALL_TENSORBASE(f)                                                                      \
  FOR_ALL_NONSCALAR_TENSORBASE(f);                                                                 \
  f(Scalar)

#define FOR_SCALAR_AND_TENSOR(f)                                                                   \
  f(Scalar);                                                                                       \
  f(Tensor)
///@}

/// @name Macros for all tensor types with comma as the separator
///@{
#define FOR_ALL_TENSORBASE_COMMA(f) FOR_ALL_PRIMITIVETENSOR_COMMA(f), f(Tensor)

#define FOR_ALL_PRIMITIVETENSOR_COMMA(f)                                                           \
  FOR_ALL_VECBASE_COMMA(f), FOR_ALL_R2BASE_COMMA(f), f(Scalar), f(SR2), f(R3), f(SFR3), f(R4),     \
      f(SFR4), f(WFR4), f(SSR4), f(SFFR4), f(SWR4), f(WSR4), f(WWR4), f(R5), f(SSFR5),             \
      f(Quaternion), f(MillerIndex), f(SSSSR8), f(R8)

#define FOR_ALL_VECBASE_COMMA(f) f(Vec), f(Rot), f(WR2)

#define FOR_ALL_R2BASE_COMMA(f) f(R2)
///@}
