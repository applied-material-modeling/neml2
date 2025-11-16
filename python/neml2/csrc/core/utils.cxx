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

#include "neml2/models/map_types.h"

#include <pybind11/pytypes.h>

#include "python/neml2/csrc/core/utils.h"

namespace py = pybind11;
using namespace neml2;

template <std::size_t D>
Tensor
unpack_tensor(
    const py::handle & pyval,
    bool assembly,
    const std::function<neml2::TensorShapeRef(const neml2::VariableName)> & base_shape_fn_i,
    const std::function<neml2::TensorShapeRef(const neml2::VariableName)> & base_shape_fn_j,
    const std::array<VariableName, D> & key)
{
  const auto key_str = key[0].str() + (D > 1 ? '/' + key[1].str() : "");

  try
  {
    const auto val = pyval.cast<Tensor>();
    if (assembly && val.base_dim() != D)
      throw py::cast_error("Invalid shape for '" + key_str + "'. Expected base dim " +
                           std::to_string(D) + " for assembly, got " +
                           std::to_string(val.base_dim()));
    return val;
  }
  catch (const py::cast_error &)
  {
    if (THPVariable_Check(pyval.ptr()))
    {
      const auto val = THPVariable_Unpack(pyval.ptr());

      if (assembly)
      {
        if (val.dim() < D)
          throw py::cast_error("Invalid shape for '" + key_str + "'. Expected at least " +
                               std::to_string(D) + " dimensions for assembly, got " +
                               utils::stringify(val.sizes()));
        return Tensor(val, val.dim() - D, 0);
      }

      if (base_shape_fn_i && base_shape_fn_j)
      {
        TensorShape base_shape{base_shape_fn_i(key[0])};
        if constexpr (D > 1)
          base_shape = utils::add_shapes(base_shape, base_shape_fn_j(key[1]));
        const auto base_dim = Size(base_shape.size());
        if (val.dim() < base_dim || val.sizes().slice(val.dim() - base_dim) != base_shape)
          throw py::cast_error("Invalid shape for '" + key_str + "'. Expected shape ending with " +
                               utils::stringify(base_shape) + ", got " +
                               utils::stringify(val.sizes()));
        return Tensor(val, val.dim() - base_dim, 0);
      }
    }
  }

  throw py::cast_error("Invalid value for '" + key_str +
                       "' -- dictionary values must be neml2.Tensor or torch.Tensor");
}

ValueMap
unpack_value_map(
    const py::dict & pyvals,
    bool assembly,
    const std::function<neml2::TensorShapeRef(const neml2::VariableName &)> & base_shape_fn)
{
  ValueMap unpacked;
  for (const auto & [pykey, pyval] : pyvals)
  {
    const auto key = pykey.cast<VariableName>();
    const auto val = unpack_tensor<1>(pyval, assembly, base_shape_fn, base_shape_fn, {key});
    unpacked[key] = val;
  }
  return unpacked;
}

DerivMap
unpack_deriv_map(
    const py::dict & pyderivs,
    bool assembly,
    const std::function<neml2::TensorShapeRef(const neml2::VariableName &)> & base_shape_fn_i,
    const std::function<neml2::TensorShapeRef(const neml2::VariableName &)> & base_shape_fn_j)
{
  DerivMap unpacked;
  for (const auto & [pykeyi, pyvals] : pyderivs)
  {
    const auto keyi = pykeyi.cast<VariableName>();

    if (!py::isinstance<py::dict>(pyvals))
      throw py::cast_error("Dictionary values must be convertible to dict");

    for (const auto & [pykeyj, pyval] : pyvals.cast<py::dict>())
    {
      const auto keyj = pykeyj.cast<VariableName>();
      const auto val =
          unpack_tensor<2>(pyval, assembly, base_shape_fn_i, base_shape_fn_j, {keyi, keyj});
      unpacked[keyi][keyj] = val;
    }
  }
  return unpacked;
}
