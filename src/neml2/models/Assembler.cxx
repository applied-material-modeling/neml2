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

#include "neml2/models/Assembler.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
Tensor
VectorAssembler::assemble_by_variable(const ValueMap & vals_dict) const
{
  const auto vars = _axis.variable_names();

  // We need to know the dtype and device so that undefined tensors can be filled with zeros
  auto options = TensorOptions();
  bool options_defined = false;

  // Look up variable values from the given dictionary.
  // If a variable is not found, tensor at that position remains undefined, and all undefined tensor
  // will later be filled with zeros.
  std::vector<Tensor> vals(vars.size());
  for (std::size_t i = 0; i < vars.size(); ++i)
  {
    const auto it = vals_dict.find(_axis.qualify(vars[i]));
    if (it != vals_dict.end())
    {
      const auto & val = it->second;
      neml_assert_dbg(val.intmd_dim() == 0,
                      "Tensor in assembly format should have no intermediate dimensions.");
      neml_assert_dbg(val.base_dim() == 1,
                      "During vector assembly, found a tensor associated with variable ",
                      vars[i],
                      " with base dimension ",
                      val.base_dim(),
                      ". Expected 1.");
      neml_assert_dbg(val.base_size(0) == _axis.variable_sizes()[i],
                      "Invalid size for variable ",
                      vars[i],
                      ". Expected ",
                      _axis.variable_sizes()[i],
                      ", got ",
                      val.base_size(0));
      neml_assert_dbg(
          val.isfinite().all().item<bool>(), "Non-finite value found for variable ", vars[i]);
      vals[i] = val;
      if (!options_defined)
      {
        options = options.dtype(vals[i].dtype()).device(vals[i].device());
        options_defined = true;
      }
    }
  }

  neml_assert(options_defined, "No variable values found for assembly");

  // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with zeros.
  const auto dynamic_sizes = utils::broadcast_dynamic_sizes(vals);
  for (std::size_t i = 0; i < vars.size(); ++i)
    if (vals[i].defined())
      vals[i] = vals[i].dynamic_expand(dynamic_sizes);
    else
      vals[i] = Tensor::zeros(dynamic_sizes, {}, _axis.variable_sizes()[i], options);

  return base_cat(vals, -1);
}

ValueMap
VectorAssembler::split_by_variable(const Tensor & tensor) const
{
  neml_assert_dbg(tensor.intmd_dim() == 0,
                  "Tensor in assembly format should have no intermediate dimensions.");
  neml_assert_dbg(tensor.base_dim() == 1,
                  "During vector split, the tensor should have a base dimension of 1.");

  ValueMap ret;

  const auto keys = _axis.variable_names();
  const auto vals = tensor.split(_axis.variable_sizes(), -1);

  for (std::size_t i = 0; i < keys.size(); ++i)
  {
    const Tensor val(vals[i], tensor.dynamic_sizes());
    ret[_axis.qualify(keys[i])] = val;
  }

  return ret;
}

std::map<SubaxisName, Tensor>
VectorAssembler::split_by_subaxis(const Tensor & tensor) const
{
  neml_assert_dbg(tensor.intmd_dim() == 0,
                  "Tensor in assembly format should have no intermediate dimensions.");
  neml_assert_dbg(tensor.base_dim() == 1,
                  "During vector split, the tensor should have a base dimension of 1.");

  std::map<SubaxisName, Tensor> ret;

  const auto keys = _axis.subaxis_names();
  const auto vals = tensor.split(_axis.subaxis_sizes(), -1);

  for (std::size_t i = 0; i < keys.size(); ++i)
    ret[_axis.qualify(keys[i])] = Tensor(vals[i], tensor.dynamic_sizes());

  return ret;
}

Tensor
MatrixAssembler::assemble_by_variable(const DerivMap & vals_dict) const
{
  const auto yvars = _yaxis.variable_names();
  const auto xvars = _xaxis.variable_names();

  // We need to know the dtype and device so that undefined tensors can be filled with zeros
  auto options = TensorOptions();
  bool options_defined = false;

  // Assemble columns of each row
  std::vector<Tensor> rows(yvars.size());
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto vals_row = vals_dict.find(_yaxis.qualify(yvars[i]));
    if (vals_row == vals_dict.end())
      continue;

    // Look up variable values from the given dictionary.
    // If a variable is not found, tensor at that position remains undefined, and all undefined
    // tensor will later be filled with zeros.
    std::vector<Tensor> vals(xvars.size());
    for (std::size_t j = 0; j < xvars.size(); ++j)
    {
      const auto it = vals_row->second.find(_xaxis.qualify(xvars[j]));
      if (it != vals_row->second.end())
      {
        const auto & val = it->second;
        neml_assert_dbg(val.intmd_dim() == 0,
                        "Tensor in assembly format should have no intermediate dimensions.");
        neml_assert_dbg(val.base_dim() == 2,
                        "During matrix assembly, found a tensor associated with variables ",
                        yvars[i],
                        "/",
                        xvars[j],
                        " with base dimension ",
                        val.base_dim(),
                        ". Expected base dimension of 2.");
        neml_assert_dbg(val.base_size(0) == _yaxis.variable_sizes()[i] &&
                            val.base_size(1) == _xaxis.variable_sizes()[j],
                        "Invalid tensor shape associated with variables ",
                        yvars[i],
                        "/",
                        xvars[j],
                        ". Expected base shape ",
                        TensorShape{_yaxis.variable_sizes()[i], _xaxis.variable_sizes()[j]},
                        ", got ",
                        val.base_sizes());
        neml_assert_dbg(val.isfinite().all().item<bool>(),
                        "Non-finite tensor value found for variables ",
                        yvars[i],
                        "/",
                        xvars[j]);
        vals[j] = val;
        if (!options_defined)
        {
          options = options.dtype(vals[j].dtype()).device(vals[j].device());
          options_defined = true;
        }
      }
    }

    neml_assert(options_defined, "No variable values found for assembly");

    // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with
    // zeros.
    const auto dynamic_sizes = utils::broadcast_dynamic_sizes(vals);
    for (std::size_t j = 0; j < xvars.size(); ++j)
      if (vals[j].defined())
        vals[j] = vals[j].dynamic_expand(dynamic_sizes);
      else
        vals[j] = Tensor::zeros(
            dynamic_sizes, {}, {_yaxis.variable_sizes()[i], _xaxis.variable_sizes()[j]}, options);

    rows[i] = base_cat(vals, -1);
  }

  // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with zeros.
  const auto dynamic_sizes = utils::broadcast_dynamic_sizes(rows);
  for (std::size_t i = 0; i < yvars.size(); ++i)
    if (rows[i].defined())
      rows[i] = rows[i].dynamic_expand(dynamic_sizes);
    else
      rows[i] =
          Tensor::zeros(dynamic_sizes, {}, {_yaxis.variable_sizes()[i], _xaxis.size()}, options);

  return base_cat(rows, -2);
}

DerivMap
MatrixAssembler::split_by_variable(const Tensor & tensor) const
{
  neml_assert_dbg(tensor.intmd_dim() == 0,
                  "Tensor in assembly format should have no intermediate dimensions.");
  neml_assert_dbg(tensor.base_dim() == 2,
                  "During vector split, the tensor should have a base dimension of 2.");

  DerivMap ret;

  const auto yvars = _yaxis.variable_names();
  const auto xvars = _xaxis.variable_names();

  const auto rows = tensor.split(_yaxis.variable_sizes(), -2);
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto vals = rows[i].split(_xaxis.variable_sizes(), -1);
    for (std::size_t j = 0; j < xvars.size(); ++j)
    {
      const Tensor val(vals[j], tensor.dynamic_sizes());
      ret[_yaxis.qualify(yvars[i])][_xaxis.qualify(xvars[j])] = val;
    }
  }

  return ret;
}

std::map<SubaxisName, std::map<SubaxisName, Tensor>>
MatrixAssembler::split_by_subaxis(const Tensor & tensor) const
{
  neml_assert_dbg(tensor.intmd_dim() == 0,
                  "Tensor in assembly format should have no intermediate dimensions.");
  neml_assert_dbg(tensor.base_dim() == 2,
                  "During vector split, the tensor should have a base dimension of 2.");

  std::map<SubaxisName, std::map<SubaxisName, Tensor>> ret;

  const auto ynames = _yaxis.subaxis_names();
  const auto xnames = _xaxis.subaxis_names();

  const auto rows = tensor.split(_yaxis.subaxis_sizes(), -2);
  for (std::size_t i = 0; i < ynames.size(); ++i)
  {
    const auto vals = rows[i].split(_xaxis.subaxis_sizes(), -1);
    for (std::size_t j = 0; j < xnames.size(); ++j)
      ret[_yaxis.qualify(ynames[i])][_xaxis.qualify(xnames[j])] =
          Tensor(vals[j], tensor.dynamic_sizes());
  }

  return ret;
}
} // namespace neml2
