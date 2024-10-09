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

#include "neml2/tensors/LabeledTensor3D.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/misc/math.h"

namespace neml2
{
void
LabeledTensor3D::fill(const LabeledTensor3D & other, bool recursive)
{
  neml_assert_dbg(axis(1) == other.axis(1), "Can only accumulate 3D tensors with conformal y axes");
  neml_assert_dbg(axis(2) == other.axis(2), "Can only accumulate 3D tensors with conformal z axes");
  const auto indices0 = axis(0).common_indices(other.axis(0), recursive);
  for (const auto & [idxi, idxi_other] : indices0)
    _tensor.base_index_put_({idxi}, other.tensor().base_index({idxi_other}));
}

LabeledTensor3D
LabeledTensor3D::assemble(std::vector<std::vector<std::vector<Tensor>>> & vals,
                          const LabeledAxis & yaxis,
                          const LabeledAxis & xaxis1,
                          const LabeledAxis & xaxis2)
{
  auto rows = std::vector<Tensor>(vals.size());

  for (std::size_t i = 0; i < vals.size(); ++i)
  {
    if (!vals[i].empty())
    {
      auto cols = std::vector<Tensor>(vals[i].size());
      for (std::size_t j = 0; j < vals[i].size(); ++j)
      {
        if (!vals[i][j].empty())
        {
          const auto batch_sizes = utils::broadcast_batch_sizes(vals[i][j]);
          const auto options = torch::TensorOptions()
                                   .dtype(utils::same_dtype(vals[i][j]))
                                   .device(utils::same_device(vals[i][j]));
          for (std::size_t k = 0; k < vals[i][j].size(); ++k)
            if (!vals[i][j][k].defined())
              vals[i][j][k] = Tensor::zeros(
                  batch_sizes,
                  {yaxis.storage_size(i), xaxis1.storage_size(j), xaxis2.storage_size(k)},
                  options);
            else
              vals[i][j][k] = vals[i][j][k].batch_expand(batch_sizes);

          cols[j] = math::base_cat(vals[i][j], -1);
        }
      }

      const auto batch_sizes = utils::broadcast_batch_sizes(cols);
      const auto options =
          torch::TensorOptions().dtype(utils::same_dtype(cols)).device(utils::same_device(cols));
      for (std::size_t j = 0; j < cols.size(); ++j)
        if (!cols[j].defined())
          cols[j] =
              Tensor::zeros(batch_sizes,
                            {yaxis.storage_size(i), xaxis1.storage_size(j), xaxis2.storage_size()},
                            options);
        else
          cols[j] = cols[j].batch_expand(batch_sizes);

      rows[i] = math::base_cat(cols, -2);
    }
  }

  const auto batch_sizes = utils::broadcast_batch_sizes(rows);
  const auto options =
      torch::TensorOptions().dtype(utils::same_dtype(rows)).device(utils::same_device(rows));
  for (std::size_t i = 0; i < rows.size(); ++i)
    if (!rows[i].defined())
      rows[i] = Tensor::zeros(batch_sizes,
                              {yaxis.storage_size(i), xaxis1.storage_size(), xaxis2.storage_size()},
                              options);
    else
      rows[i] = rows[i].batch_expand(batch_sizes);

  return LabeledTensor3D(math::base_cat(rows, -3), {&yaxis, &xaxis1, &xaxis2});
}
} // namespace neml2
