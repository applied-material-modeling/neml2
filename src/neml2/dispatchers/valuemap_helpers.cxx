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
#include "neml2/misc/assertions.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{

ValueMap
valuemap_cat_reduce(std::vector<ValueMap> && results, Size dynamic_dim)
{
  // Figure out the dispatch size for each dispatch
  std::vector<Size> dispatch_sizes(results.size());
  for (size_t i = 0; i < results.size(); ++i)
  {
    for (auto && [name, value] : results[i])
      if (dynamic_dim >= -value.dynamic_dim() && dynamic_dim < value.dynamic_dim())
      {
        const auto si = value.dynamic_size(dynamic_dim).concrete();
        if (dispatch_sizes[i] == 0)
          dispatch_sizes[i] = si;
        else
          neml_assert(dispatch_sizes[i] == si,
                      "Inconsistent dispatch sizes within dispatch ",
                      i,
                      ": got ",
                      dispatch_sizes[i],
                      " and ",
                      si,
                      " for variable ",
                      name,
                      ".");
      }
      else
        neml_assert(value.dynamic_dim() == 0,
                    "Reduction along dynamic dimension ",
                    dynamic_dim,
                    " does not make sense for variable ",
                    name,
                    " with dynamic shape ",
                    value.dynamic_sizes());
    neml_assert(
        dispatch_sizes[i] > 0, "Unable to determine the dispatch size for dispatch ", i, ".");
  }

  // Re-bin the results, broadcasting as needed
  std::map<VariableName, std::vector<Tensor>> vars;
  for (std::size_t i = 0; i < results.size(); ++i)
    for (auto && [name, value] : results[i])
    {
      if (dynamic_dim >= -value.dynamic_dim() && dynamic_dim < value.dynamic_dim())
        vars[name].emplace_back(std::move(value));
      else
        vars[name].push_back(value.dynamic_expand(dispatch_sizes[i]));
    }

  // Concatenate the tensors
  ValueMap ret;
  for (auto && [name, values] : vars)
    ret[name] = dynamic_cat(values, dynamic_dim);

  return ret;
}

ValueMap
valuemap_move_device(ValueMap && x, Device device)
{
  // Move the tensors to the device
  for (auto && [name, value] : std::move(x))
    x[name] = value.to(device);
  return std::move(x);
}

ValueMap
valuemap_no_operation(ValueMap && x)
{
  return std::move(x);
}

}
