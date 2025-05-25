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
#include <torch/csrc/autograd/python_variable_indexing.h>

namespace pybind11::detail
{
/**
 * @brief Type conversion between Python object <--> at::indexing::Slice
 */
template <>
struct type_caster<at::indexing::Slice>
{
public:
  PYBIND11_TYPE_CASTER(at::indexing::Slice, const_name("Any"));

  bool load(handle src, bool)
  {
    PyObject * slice = src.ptr();

    if (!PySlice_Check(slice))
      return false;

    const auto val = torch::autograd::__PySlice_Unpack(slice);
    value = at::indexing::Slice(val.start, val.stop, val.step);
    return true;
  }

  static handle
  cast(const at::indexing::Slice & src, return_value_policy /* policy */, handle /* parent */)
  {
    auto start = THPUtils_packInt64(src.start().expect_int());
    auto stop = THPUtils_packInt64(src.stop().expect_int());
    auto step = THPUtils_packInt64(src.step().expect_int());
    return PySlice_New(start, stop, step);
  }
};

/**
 * @brief Type conversion between Python object <--> TensorIndex
 */
template <>
struct type_caster<at::indexing::TensorIndex>
{
public:
  PYBIND11_TYPE_CASTER(at::indexing::TensorIndex, const_name("Any"));

  /**
   * PYBIND11_TYPE_CASTER defines a member field called value. Since at::indexing::TensorIndex
   * cannot be default-initialized, we provide this constructor to explicitly initialize that field.
   * The value doesn't matter as it will be overwritten after a successful call to load.
   */
  type_caster()
    : value(at::indexing::None)
  {
  }

  bool load(handle src, bool)
  {
    PyObject * index = src.ptr();

    // handle simple types: none, ellipsis
    if (index == Py_None)
    {
      value = at::indexing::None;
      return true;
    }
    if (index == Py_Ellipsis)
    {
      value = at::indexing::Ellipsis;
      return true;
    }

    // handle simple types: integers, slices, bool
    if (THPUtils_checkLong(index))
    {
      value = at::indexing::TensorIndex(THPUtils_unpackLong(index));
      return true;
    }
    if (PySlice_Check(index))
    {
      auto val = torch::autograd::__PySlice_Unpack(index);
      value = at::indexing::TensorIndex(at::indexing::Slice(val.start, val.stop, val.step));
      return true;
    }
    if (index == Py_False || index == Py_True)
    {
      value = at::indexing::TensorIndex(index == Py_True);
      return true;
    }

    // TODO: indexing by tensors ("advanced" indexing)

    return false;
  }

  static handle
  cast(const at::indexing::TensorIndex & src, return_value_policy /* policy */, handle /* parent */)
  {
    if (src.is_none())
      return Py_None;
    if (src.is_ellipsis())
      return Py_Ellipsis;
    if (src.is_integer())
      // Will we ever support SymInt? I don't think so...
      return THPUtils_packInt64(src.integer().expect_int());
    if (src.is_slice())
    {
      auto start = THPUtils_packInt64(src.slice().start().expect_int());
      auto stop = THPUtils_packInt64(src.slice().stop().expect_int());
      auto step = THPUtils_packInt64(src.slice().step().expect_int());
      return PySlice_New(start, stop, step);
    }
    if (src.is_boolean())
      return src.boolean() ? Py_True : Py_False;
    return {};
  }
};
}
