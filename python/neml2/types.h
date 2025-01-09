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

#include <torch/python.h>
#include <torch/version.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/DynamicTypes.h>

#include "neml2/misc/types.h"
#include "neml2/base/LabeledAxisAccessor.h"

#define NEML2_TENSOR_OPTIONS_VARGS                                                                 \
  const torch::Dtype &dtype, const torch::Device &device, bool requires_grad

#define NEML2_TENSOR_OPTIONS                                                                       \
  torch::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad)

#define PY_ARG_TENSOR_OPTIONS                                                                      \
  pybind11::arg("dtype") = torch::Dtype(torch::kFloat64),                                          \
  pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("requires_grad") = false

namespace pybind11
{
namespace detail
{
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)
/**
 * @brief This specialization enables type conversion between Python object <--> torch::Dtype
 */
template <>
struct type_caster<torch::Dtype>
{
public:
  PYBIND11_TYPE_CASTER(torch::Dtype, _("torch.dtype"));

  /**
   * PYBIND11_TYPE_CASTER defines a member field called value. Since at::Dtype cannot be
   * default-initialized, we provide this constructor to explicitly initialize that field. The value
   * doesn't matter as it will be overwritten after a successful call to load.
   */
  type_caster()
    : value(torch::kFloat64)
  {
  }

  bool load(handle src, bool)
  {
    PyObject * obj = src.ptr();
    if (THPDtype_Check(obj))
    {
      value = reinterpret_cast<THPDtype *>(obj)->scalar_type;
      return true;
    }
    return false;
  }

  static handle
  cast(const torch::Dtype & src, return_value_policy /* policy */, handle /* parent */)
  {
    return handle(reinterpret_cast<PyObject *>(torch::getTHPDtype(src)));
  }
};
#endif

/**
 * @brief This specialization exposes neml2::indexing::TensorIndices
 */
template <>
struct type_caster<neml2::indexing::TensorIndices>
{
public:
  PYBIND11_TYPE_CASTER(neml2::indexing::TensorIndices, const_name("list[Any]"));

  bool load(handle src, bool)
  {
    // if src is an iterable
    if (isinstance<iterable>(src))
    {
      auto src_iterable = reinterpret_borrow<iterable>(src);
      for (auto item : src_iterable)
        value.push_back(item.cast<neml2::indexing::TensorIndex>());
      return true;
    }

    return false;
  }

  static handle cast(const neml2::indexing::TensorIndices & src,
                     return_value_policy /* policy */,
                     handle /* parent */)
  {
    list l;
    for (const auto & val : src)
      l.append(val);
    return l;
  }
};
}
}
