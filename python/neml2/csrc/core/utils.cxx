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
#include "neml2/models/Model.h"

#include "python/neml2/csrc/core/utils.h"

namespace py = pybind11;
using namespace neml2;

ValueMap
unpack_tensor_map(const py::dict & pyinputs, const Model * model)
{
  std::vector<VariableName> input_names;
  std::vector<Tensor> input_values;
  for (auto && [key, val] : pyinputs)
  {
    try
    {
      input_names.push_back(key.cast<VariableName>());
    }
    catch (py::cast_error &)
    {
      throw py::cast_error("Dictionary keys must be convertible to neml2.VariableName");
    }

    try
    {
      input_values.push_back(val.cast<Tensor>());
    }
    catch (py::cast_error &)
    {
      throw py::cast_error("Invalid value for variable '" + input_names.back().str() +
                           "' -- dictionary values must be convertible to neml2.Tensor");
    }
  }

  ValueMap inputs;
  for (size_t i = 0; i < input_names.size(); ++i)
    inputs[input_names[i]] = input_values[i];

  return inputs;
}
