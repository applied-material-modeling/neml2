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
#include "neml2/base/Parser.h"
#include "neml2/tensors/shape_utils.h"

#include <pybind11/pytypes.h>

#include "csrc/core/utils.h"

namespace py = pybind11;
using namespace neml2;

template <std::size_t D>
static Tensor
unpack_tensor(const py::handle & pyval,
              bool assembly,
              const std::array<TensorShapeRef, D> & base_shapes)
{
  try
  {
    const auto val = pyval.cast<Tensor>();
    if (assembly && val.base_dim() != D)
      throw py::value_error("Invalid shape. Expected base dim " + std::to_string(D) +
                            " for assembly, got " + std::to_string(val.base_dim()));
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
          throw py::value_error("Invalid shape. Expected at least " + std::to_string(D) +
                                " dimensions for assembly, got " + utils::stringify(val.sizes()));
        return Tensor(val, val.dim() - D, 0);
      }

      auto base_shape = TensorShape(base_shapes[0]);
      if constexpr (D > 1)
        base_shape = utils::add_shapes(base_shape, base_shapes[1]);
      const auto base_dim = Size(base_shape.size());
      if (val.dim() < base_dim || val.sizes().slice(val.dim() - base_dim) != base_shape)
        throw py::value_error("Invalid shape. Expected shape ending with " +
                              utils::stringify(base_shape) + ", got " +
                              utils::stringify(val.sizes()));
      return Tensor(val, val.dim() - base_dim, 0);
    }
  }

  throw py::value_error(
      "Invalid value tensor -- dictionary values must be neml2.Tensor or torch.Tensor");
}

template <std::size_t D>
static Tensor
unpack_tensor(const py::handle & pyval,
              bool assembly,
              const std::function<TensorShapeRef(const VariableName &)> & base_shape_fn_i,
              const std::function<TensorShapeRef(const VariableName &)> & base_shape_fn_j,
              const std::array<VariableName, D> & key)
{
  const auto key_str = key[0].str() + (D > 1 ? '/' + key[1].str() : "");
  std::array<TensorShapeRef, D> base_shapes;
  base_shapes[0] = base_shape_fn_i ? base_shape_fn_i(key[0]) : TensorShapeRef{};
  if constexpr (D > 1)
    base_shapes[1] = base_shape_fn_j ? base_shape_fn_j(key[1]) : TensorShapeRef{};
  return unpack_tensor<D>(pyval, assembly, base_shapes);
}

std::vector<neml2::Tensor>
unpack_tensor_list(const pybind11::sequence & pytensors,
                   const std::function<neml2::TensorShapeRef(std::size_t)> & base_shape_fn)
{
  std::vector<neml2::Tensor> unpacked;
  unpacked.reserve(pytensors.size());
  for (std::size_t i = 0; i < pytensors.size(); ++i)
  {
    const auto tensor = unpack_tensor<1>(pytensors[i], false, {base_shape_fn(i)});
    unpacked.push_back(tensor);
  }
  return unpacked;
}

ValueMap
unpack_value_map(const py::dict & pyvals,
                 bool assembly,
                 const std::function<TensorShapeRef(const VariableName &)> & base_shape_fn)
{
  ValueMap unpacked;
  for (const auto & [pykey, pyval] : pyvals)
  {
    const auto key = unpack_variable_name(pykey);
    const auto val = unpack_tensor<1>(pyval, assembly, base_shape_fn, base_shape_fn, {key});
    unpacked[key] = val;
  }
  return unpacked;
}

DerivMap
unpack_deriv_map(const py::dict & pyderivs,
                 bool assembly,
                 const std::function<TensorShapeRef(const VariableName &)> & base_shape_fn_i,
                 const std::function<TensorShapeRef(const VariableName &)> & base_shape_fn_j)
{
  DerivMap unpacked;
  for (const auto & [pykeyi, pyvals] : pyderivs)
  {
    const auto keyi = unpack_variable_name(pykeyi);

    if (!py::isinstance<py::dict>(pyvals))
      throw py::value_error("Dictionary values must be convertible to dict");

    for (const auto & [pykeyj, pyval] : pyvals.cast<py::dict>())
    {
      const auto keyj = unpack_variable_name(pykeyj);
      const auto val =
          unpack_tensor<2>(pyval, assembly, base_shape_fn_i, base_shape_fn_j, {keyi, keyj});
      unpacked[keyi][keyj] = val;
    }
  }
  return unpacked;
}

std::map<std::string, Tensor>
pack_value_map(const ValueMap & vals)
{
  std::map<std::string, Tensor> dict;
  for (const auto & [key, val] : vals)
    dict[key.str()] = val;
  return dict;
}

std::map<std::string, std::map<std::string, Tensor>>
pack_deriv_map(const DerivMap & derivs)
{
  std::map<std::string, std::map<std::string, Tensor>> dict;
  for (const auto & [keyi, vals] : derivs)
    dict[keyi.str()] = pack_value_map(vals);
  return dict;
}

VariableName
unpack_variable_name(const py::handle & obj)
{
  if (py::isinstance<py::str>(obj))
    return utils::parse<VariableName>(obj.cast<std::string>());
  else if (py::isinstance<VariableName>(obj))
    return obj.cast<VariableName>();
  else
    throw py::cast_error("Cannot convert object to VariableName");
}
