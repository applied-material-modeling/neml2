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

#include "neml2/dispatchers/valuemap_helpers.h"
#include "neml2/tensors/functions/cat.h"

namespace neml2
{

ValueMap
valuemap_cat_reduce(std::vector<ValueMap> && results, Size batch_dim)
{
  // Re-bin the results
  std::map<VariableName, std::vector<Tensor>> vars;
  for (auto && result : results)
    for (auto && [name, value] : result)
      vars[name].emplace_back(std::move(value));

  // Concatenate the tensors
  ValueMap ret;
  for (auto && [name, values] : vars)
    ret[name] = batch_cat(values, batch_dim);

  return ret;
}

ValueMap
valuemap_move_device(ValueMap && x, Device device)
{
  // Move the tensors to the device
  for (auto && [name, value] : x)
    x[name] = value.to(device);
  return x;
}

ValueMap
valuemap_no_operation(ValueMap && x)
{
  return std::move(x);
}

}
