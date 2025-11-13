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

#define NEML2_TENSOR_OPTIONS_VARGS const Dtype &dtype, const Device &device, bool requires_grad

#define NEML2_TENSOR_OPTIONS                                                                       \
  torch::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad)

#define PY_ARG_TENSOR_OPTIONS                                                                      \
  pybind11::arg("dtype") = Dtype(torch::kFloat64), pybind11::arg("device") = Device(torch::kCPU),  \
  pybind11::arg("requires_grad") = false

#define DEFAULT_TYPECASTER(cpptype, pyname)                                                        \
  namespace pybind11::detail                                                                       \
  {                                                                                                \
  template <>                                                                                      \
  struct type_caster<cpptype> : type_caster_base<cpptype>                                          \
  {                                                                                                \
    static constexpr auto name = const_name(pyname);                                               \
  };                                                                                               \
  }                                                                                                \
  static_assert(true)
