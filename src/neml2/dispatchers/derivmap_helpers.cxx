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

#include "neml2/dispatchers/derivmap_helpers.h"
#include "neml2/tensors/functions/cat.h"

namespace neml2
{

DerivMap
derivmap_cat_reduce(std::vector<DerivMap> && results, Size batch_dim)
{
  // Re-bin the results
  std::map<VariableName, std::map<VariableName, std::vector<Tensor>>> vars;
  for (auto && result : results)
    for (auto && [name1, vmap] : result)
      for (auto && [name2, value] : vmap)
        vars[name1][name2].emplace_back(std::move(value));

  // Concatenate the tensors
  DerivMap ret;
  for (auto && [name1, vmap] : vars)
    for (auto && [name2, values] : vmap)
      ret[name1][name2] = batch_cat(values, batch_dim);

  return ret;
}

DerivMap
derivmap_move_device(DerivMap && x, Device device)
{
  // Move the tensors to the device
  for (auto && [name, vmap] : x)
    for (auto && [name2, value] : vmap)
      x[name][name2] = value.to(device);
  return x;
}

DerivMap
derivmap_no_operation(DerivMap && x)
{
  return std::move(x);
}

}
