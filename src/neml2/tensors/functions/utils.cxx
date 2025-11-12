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

#include "neml2/tensors/functions/utils.h"

namespace neml2::utils
{
std::pair<std::vector<Tensor>, Size>
align_intmd_dim(TensorList tensors)
{
  Size dmax = 0;
  for (const auto & tensor : tensors)
    if (tensor.intmd_dim() > dmax)
      dmax = tensor.intmd_dim();

  std::vector<Tensor> aligned(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i)
    aligned[i] = tensors[i].intmd_unsqueeze_n(dmax - tensors[i].intmd_dim(), 0);
  return {aligned, dmax};
}

std::pair<std::vector<Tensor>, Size>
align_static_dim(TensorList tensors)
{
  Size imax = 0;
  Size bmax = 0;
  for (const auto & tensor : tensors)
  {
    if (tensor.intmd_dim() > imax)
      imax = tensor.intmd_dim();
    if (tensor.base_dim() > bmax)
      bmax = tensor.base_dim();
  }

  std::vector<Tensor> aligned(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i)
    aligned[i] = tensors[i]
                     .intmd_unsqueeze_n(imax - tensors[i].intmd_dim(), 0)
                     .base_unsqueeze_n(bmax - tensors[i].base_dim(), 0);
  return {aligned, imax};
}
}
