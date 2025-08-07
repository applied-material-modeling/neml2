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
#include "neml2/jit/utils.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"
#include <numeric>

namespace neml2
{
namespace utils
{
template <std::size_t N>
Tensor
to_assembly(const Tensor & from,
            const std::array<TensorShapeRef, N> & base_sizes,
            const std::array<TensorShapeRef, N> & lbatch_sizes,
            const std::string & debug_name)
{
#ifndef NDEBUG
  // check if the given tensor has the correct left-batch shape
  Size offset = 0;
  for (std::size_t i = 0; i < N; i++)
  {
    auto lbatch_dim = Size(lbatch_sizes[i].size());
    neml_assert_dbg((from.batch_dim() - offset) >= lbatch_dim,
                    "Insufficient batch dimension for left-batch ",
                    i,
                    " of tensor '",
                    debug_name,
                    "', left-batch dimension is ",
                    lbatch_dim,
                    ", remaining batch dimension is ",
                    (from.batch_dim() - offset));
    neml_assert_dbg(from.batch_sizes().slice(offset, lbatch_dim) == lbatch_sizes[i],
                    "Incompatible left-batch shape for group ",
                    i,
                    " of tensor '",
                    debug_name,
                    "', expected left-batch shape is ",
                    lbatch_sizes[i],
                    ", but got ",
                    from.batch_sizes().slice(offset, lbatch_dim),
                    ".");
    offset += lbatch_dim;
  }

  // check if the given tensor has the correct base shape
  offset = 0;
  for (std::size_t i = 0; i < N; i++)
  {
    auto base_dim = Size(base_sizes[i].size());
    neml_assert_dbg((from.base_dim() - offset) >= base_dim,
                    "Insufficient batch dimension for base ",
                    i,
                    " of tensor '",
                    debug_name,
                    "', base dimension is ",
                    base_dim,
                    ", remaining batch dimension is ",
                    (from.base_dim() - offset));
    neml_assert_dbg(from.base_sizes().slice(offset, base_dim) == base_sizes[i],
                    "Incompatible base shape for group ",
                    i,
                    " of tensor '",
                    debug_name,
                    "', expected base shape is ",
                    base_sizes[i],
                    ", but got ",
                    from.base_sizes().slice(offset, base_dim),
                    ".");
    offset += base_dim;
  }
#endif

  // Generate permutation to move each left-batch to the base
  //
  // For example, for N == 2, the tensor shape is in the form of
  //   (lbatch1, lbatch2, batch; base1, base2)
  //
  // We first move lbatch1 before base1:
  //   (lbatch2, batch; lbatch1, base1, base2)
  //
  // Then we move lbatch2 before base2:
  //   (batch; lbatch1, base1, lbatch2, base2)
  Size lbatch_dim = 0;
  TensorShape indices(from.dim());
  TensorShape assembly_sizes(N);
  std::iota(indices.begin(), indices.end(), 0);
  auto permutation = indices;
  auto first = permutation.begin();
  auto last = first + from.batch_dim();
  for (std::size_t i = 0; i < N; ++i)
  {
    assembly_sizes[i] = utils::storage_size(lbatch_sizes[i]) * utils::storage_size(base_sizes[i]);
    if (!lbatch_sizes[i].empty())
    {
      lbatch_dim += Size(lbatch_sizes[i].size());
      auto middle = first + lbatch_sizes[i].size();
      std::rotate(first, middle, last);
    }
    last += base_sizes[i].size();
  }

  // Perform the permutation
  auto B = from.batch_sizes().slice(lbatch_dim);
  auto permuted = Tensor(at::permute(from, permutation), B);
  return permuted.base_reshape(assembly_sizes);
}

// Explicit instantiations
template Tensor to_assembly<1>(const Tensor &,
                               const std::array<TensorShapeRef, 1> &,
                               const std::array<TensorShapeRef, 1> &,
                               const std::string &);
template Tensor to_assembly<2>(const Tensor &,
                               const std::array<TensorShapeRef, 2> &,
                               const std::array<TensorShapeRef, 2> &,
                               const std::string &);
template Tensor to_assembly<3>(const Tensor &,
                               const std::array<TensorShapeRef, 3> &,
                               const std::array<TensorShapeRef, 3> &,
                               const std::string &);

template <std::size_t N>
Tensor
from_assembly(const Tensor & from,
              const std::array<TensorShapeRef, N> & base_sizes,
              const std::array<TensorShapeRef, N> & lbatch_sizes,
              const std::string & debug_name)
{
#ifndef NDEBUG
  TensorShape assembly_sizes(N);
  for (std::size_t i = 0; i < N; i++)
    assembly_sizes[i] = utils::storage_size(lbatch_sizes[i]) * utils::storage_size(base_sizes[i]);
  neml_assert_dbg(assembly_sizes == from.base_sizes(),
                  "Incompatible base shape for tensor '",
                  debug_name,
                  "', expected base shape is ",
                  assembly_sizes,
                  ", but got ",
                  from.base_sizes());
#endif

  // Generate the unflattened base shape and left-batch shape
  TensorShape unfl_sizes, total_lbatch_sizes;
  for (std::size_t i = 0; i < N; i++)
  {
    unfl_sizes.insert(unfl_sizes.end(), lbatch_sizes[i].begin(), lbatch_sizes[i].end());
    unfl_sizes.insert(unfl_sizes.end(), base_sizes[i].begin(), base_sizes[i].end());
    total_lbatch_sizes.insert(
        total_lbatch_sizes.end(), lbatch_sizes[i].begin(), lbatch_sizes[i].end());
  }

  // Unflatten base
  auto unfl = from.base_reshape(unfl_sizes);

  // Generate permutation to move each left-batch to the batch
  //
  // For example, for N == 2, the tensor shape is in the form of
  //   (batch; lbatch1, base1, lbatch2, base2)
  //
  // We first move lbatch1 to the front:
  //   (lbatch1, batch; base1, lbatch2, base2)
  //
  // Then we move lbatch2 to the front:
  //   (lbatch1, lbatch2, batch; base1, base2)
  TensorShape indices(unfl.dim());
  std::iota(indices.begin(), indices.end(), 0);
  auto permutation = indices;
  auto first = permutation.begin();
  auto middle = first + unfl.batch_dim();
  for (std::size_t i = 0; i < N; ++i)
  {
    if (!lbatch_sizes[i].empty())
    {
      auto last = middle + lbatch_sizes[i].size();
      std::rotate(first, middle, last);
      middle += lbatch_sizes[i].size();
    }
    middle += base_sizes[i].size();
  }

  // Perform the permutation
  auto B = utils::add_traceable_shapes(total_lbatch_sizes, from.batch_sizes());
  return Tensor(at::permute(unfl, permutation), B);
}

// Explicit instantiations
template Tensor from_assembly<1>(const Tensor &,
                                 const std::array<TensorShapeRef, 1> &,
                                 const std::array<TensorShapeRef, 1> &,
                                 const std::string &);
template Tensor from_assembly<2>(const Tensor &,
                                 const std::array<TensorShapeRef, 2> &,
                                 const std::array<TensorShapeRef, 2> &,
                                 const std::string &);
template Tensor from_assembly<3>(const Tensor &,
                                 const std::array<TensorShapeRef, 3> &,
                                 const std::array<TensorShapeRef, 3> &,
                                 const std::string &);

} // namespace utils

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
      neml_assert_dbg(val.base_dim() == 1,
                      "During matrix assembly, found a tensor associated with variable ",
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
      vals[i] = val;
      if (!options_defined)
      {
        options = options.dtype(vals[i].dtype()).device(vals[i].device());
        options_defined = true;
      }
    }
  }

  neml_assert(options_defined, "No variable values found for assembly");

  // Expand defined tensors with the broadcast batch shape and fill undefined tensors with zeros.
  const auto batch_sizes = utils::broadcast_batch_sizes(vals);
  for (std::size_t i = 0; i < vars.size(); ++i)
    if (vals[i].defined())
      vals[i] = vals[i].batch_expand(batch_sizes);
    else
      vals[i] = Tensor::zeros(batch_sizes, _axis.variable_sizes()[i], options);

  return base_cat(vals, -1);
}

ValueMap
VectorAssembler::split_by_variable(const Tensor & tensor) const
{
  ValueMap ret;

  const auto keys = _axis.variable_names();
  const auto vals = tensor.split(_axis.variable_sizes(), -1);

  for (std::size_t i = 0; i < keys.size(); ++i)
  {
    const Tensor val(vals[i], tensor.batch_sizes());
    ret[_axis.qualify(keys[i])] = val;
  }

  return ret;
}

std::map<SubaxisName, Tensor>
VectorAssembler::split_by_subaxis(const Tensor & tensor) const
{
  std::map<SubaxisName, Tensor> ret;

  const auto keys = _axis.subaxis_names();
  const auto vals = tensor.split(_axis.subaxis_sizes(), -1);

  for (std::size_t i = 0; i < keys.size(); ++i)
    ret[_axis.qualify(keys[i])] = Tensor(vals[i], tensor.batch_sizes());

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
        vals[j] = val;
        if (!options_defined)
        {
          options = options.dtype(vals[j].dtype()).device(vals[j].device());
          options_defined = true;
        }
      }
    }

    neml_assert(options_defined, "No variable values found for assembly");

    // Expand defined tensors with the broadcast batch shape and fill undefined tensors with zeros.
    const auto batch_sizes = utils::broadcast_batch_sizes(vals);
    for (std::size_t j = 0; j < xvars.size(); ++j)
      if (vals[j].defined())
        vals[j] = vals[j].batch_expand(batch_sizes);
      else
        vals[j] = Tensor::zeros(
            batch_sizes, {_yaxis.variable_sizes()[i], _xaxis.variable_sizes()[j]}, options);

    rows[i] = base_cat(vals, -1);
  }

  // Expand defined tensors with the broadcast batch shape and fill undefined tensors with zeros.
  const auto batch_sizes = utils::broadcast_batch_sizes(rows);
  for (std::size_t i = 0; i < yvars.size(); ++i)
    if (rows[i].defined())
      rows[i] = rows[i].batch_expand(batch_sizes);
    else
      rows[i] = Tensor::zeros(batch_sizes, {_yaxis.variable_sizes()[i], _xaxis.size()}, options);

  return base_cat(rows, -2);
}

DerivMap
MatrixAssembler::split_by_variable(const Tensor & tensor) const
{
  DerivMap ret;

  const auto yvars = _yaxis.variable_names();
  const auto xvars = _xaxis.variable_names();

  const auto rows = tensor.split(_yaxis.variable_sizes(), -2);
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto vals = rows[i].split(_xaxis.variable_sizes(), -1);
    for (std::size_t j = 0; j < xvars.size(); ++j)
    {
      const Tensor val(vals[j], tensor.batch_sizes());
      ret[_yaxis.qualify(yvars[i])][_xaxis.qualify(xvars[j])] = val;
    }
  }

  return ret;
}

std::map<SubaxisName, std::map<SubaxisName, Tensor>>
MatrixAssembler::split_by_subaxis(const Tensor & tensor) const
{
  std::map<SubaxisName, std::map<SubaxisName, Tensor>> ret;

  const auto ynames = _yaxis.subaxis_names();
  const auto xnames = _xaxis.subaxis_names();

  const auto rows = tensor.split(_yaxis.subaxis_sizes(), -2);
  for (std::size_t i = 0; i < ynames.size(); ++i)
  {
    const auto vals = rows[i].split(_xaxis.subaxis_sizes(), -1);
    for (std::size_t j = 0; j < xnames.size(); ++j)
      ret[_yaxis.qualify(ynames[i])][_xaxis.qualify(xnames[j])] =
          Tensor(vals[j], tensor.batch_sizes());
  }

  return ret;
}
} // namespace neml2
