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

#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/misc/math.h"

namespace neml2
{
LabeledMatrix
LabeledMatrix::identity(TensorShapeRef batch_size,
                        const LabeledAxis & axis,
                        const torch::TensorOptions & options)
{
  return LabeledMatrix(Tensor::identity(batch_size, axis.storage_size(), options), {&axis, &axis});
}

void
LabeledMatrix::fill(const LabeledMatrix & other, bool recursive)
{
  neml_assert_dbg(axis(1) == other.axis(1), "Can only accumulate matrices with conformal y axes");
  const auto indices = axis(0).common_indices(other.axis(0), recursive);
  for (const auto & [idxi, idxi_other] : indices)
    _tensor.base_index_put_({idxi}, other.tensor().base_index({idxi_other}));
}

std::map<LabeledAxisAccessor, Tensor>
LabeledMatrix::split_variables(Size dim, bool qualified) const
{
  std::map<LabeledAxisAccessor, Tensor> ret;

  const auto vars = qualified ? axis(dim).qualified_variable_names() : axis(dim).variable_names();
  const auto vals = tensor().split(axis(dim).variable_sizes(), batch_dim() + dim);
  for (std::size_t i = 0; i < vars.size(); ++i)
    ret[vars[i]] = Tensor(vals[i], batch_sizes());

  return ret;
}

std::map<LabeledAxisAccessor, LabeledMatrix>
LabeledMatrix::split_subaxes(Size dim, bool qualified) const
{
  std::map<LabeledAxisAccessor, LabeledMatrix> ret;

  const auto keys = qualified ? axis(dim).qualified_subaxis_names() : axis(dim).subaxis_names();
  const auto subaxes = axis(dim).subaxis_names();
  const auto vals = tensor().split(axis(dim).subaxis_sizes(), batch_dim() + dim);
  for (std::size_t i = 0; i < keys.size(); ++i)
    ret[keys[i]] = LabeledMatrix(Tensor(vals[i], batch_sizes()),
                                 {dim == 0 ? &axis(0).subaxis(subaxes[i]) : &axis(0),
                                  dim == 1 ? &axis(1).subaxis(subaxes[i]) : &axis(1)});

  return ret;
}

std::map<LabeledAxisAccessor, std::map<LabeledAxisAccessor, Tensor>>
LabeledMatrix::disassemble_variables(bool qualified) const
{
  std::map<LabeledAxisAccessor, std::map<LabeledAxisAccessor, Tensor>> ret;

  const auto yvars = qualified ? axis(0).qualified_variable_names() : axis(0).variable_names();
  const auto xvars = qualified ? axis(1).qualified_variable_names() : axis(1).variable_names();
  const auto rows = tensor().split(axis(0).variable_sizes(), -2);
  for (std::size_t i = 0; i < rows.size(); ++i)
  {
    const auto vals = rows[i].split(axis(1).variable_sizes(), -1);
    for (std::size_t j = 0; j < vals.size(); ++j)
      ret[yvars[i]][xvars[j]] = Tensor(vals[j], batch_sizes());
  }

  return ret;
}

LabeledMatrix
LabeledMatrix::assemble(std::vector<std::vector<Tensor>> & vals,
                        const LabeledAxis & yaxis,
                        const LabeledAxis & xaxis)
{
  auto rows = std::vector<Tensor>(vals.size());

  // Assemble columns
  for (std::size_t i = 0; i < vals.size(); ++i)
    if (!vals[i].empty())
    {
      const auto batch_sizes = utils::broadcast_batch_sizes(vals[i]);
      const auto options = torch::TensorOptions()
                               .dtype(utils::same_dtype(vals[i]))
                               .device(utils::same_device(vals[i]));
      for (std::size_t j = 0; j < vals[i].size(); ++j)
        if (!vals[i][j].defined())
          vals[i][j] =
              Tensor::zeros(batch_sizes, {yaxis.storage_size(i), xaxis.storage_size(j)}, options);
        else
          vals[i][j] = vals[i][j].batch_expand(batch_sizes);

      rows[i] = math::base_cat(vals[i], -1);
    }

  // Assemble rows
  const auto batch_sizes = utils::broadcast_batch_sizes(rows);
  const auto options =
      torch::TensorOptions().dtype(utils::same_dtype(rows)).device(utils::same_device(rows));
  for (std::size_t i = 0; i < rows.size(); ++i)
    if (!rows[i].defined())
      rows[i] = Tensor::zeros(batch_sizes, {yaxis.storage_size(i), xaxis.storage_size()}, options);
    else
      rows[i] = rows[i].batch_expand(batch_sizes);

  return LabeledMatrix(math::base_cat(rows, -2), {&yaxis, &xaxis});
}
} // namespace neml2
