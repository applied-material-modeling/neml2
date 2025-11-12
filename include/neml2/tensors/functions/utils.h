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

#include <tuple>
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/Tensor.h"

namespace neml2::utils
{
template <typename T, typename... Ts>
std::tuple<T, Ts..., Size>
align_intmd_dim(const T & a, const Ts &... ts)
{
  neml_assert_intmd_broadcastable_dbg(a, ts...);

  if ((... && (a.intmd_dim() == ts.intmd_dim())))
    return {a, ts..., a.intmd_dim()};

  const auto dmax = utils::broadcast_intmd_dim(a, ts...);
  return {a.intmd_unsqueeze_n(dmax - a.intmd_dim(), 0),
          ts.intmd_unsqueeze_n(dmax - ts.intmd_dim(), 0)...,
          dmax};
}

std::pair<std::vector<Tensor>, Size> align_intmd_dim(TensorList tensors);

template <typename T, typename... Ts>
std::tuple<T, Ts..., Size>
align_static_dim(const T & a, const Ts &... ts)
{
  neml_assert_static_broadcastable_dbg(a, ts...);

  if ((... && (a.intmd_dim() == ts.intmd_dim() && a.base_dim() == ts.base_dim())))
    return {a, ts..., a.intmd_dim()};

  const auto imax = utils::broadcast_intmd_dim(a, ts...);
  const auto bmax = utils::broadcast_base_dim(a, ts...);
  return {
      a.intmd_unsqueeze_n(imax - a.intmd_dim(), 0).base_unsqueeze_n(bmax - a.base_dim(), 0),
      ts.intmd_unsqueeze_n(imax - ts.intmd_dim(), 0).base_unsqueeze_n(bmax - ts.base_dim(), 0)...,
      imax};
}

std::pair<std::vector<Tensor>, Size> align_static_dim(TensorList tensors);
}
