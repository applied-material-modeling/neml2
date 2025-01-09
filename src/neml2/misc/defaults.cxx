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

#include <torch/types.h>

#include "neml2/misc/types.h"
#include "neml2/misc/defaults.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
torch::TensorOptions &
default_tensor_options()
{
  static torch::TensorOptions _default_tensor_options =
      torch::TensorOptions().dtype(default_dtype()).device(default_device());
  return _default_tensor_options;
}

torch::TensorOptions &
default_integer_tensor_options()
{
  static torch::TensorOptions _default_integer_tensor_options =
      torch::TensorOptions().dtype(default_integer_dtype()).device(default_device());
  return _default_integer_tensor_options;
}

torch::Dtype &
default_dtype()
{
  static torch::Dtype _default_dtype = torch::kFloat64;
  return _default_dtype;
}

torch::Dtype &
default_integer_dtype()
{
  static torch::Dtype _default_integer_dtype = torch::kInt64;
  return _default_integer_dtype;
}

torch::Device &
default_device()
{
  static torch::Device _default_device = torch::kCPU;
  return _default_device;
}

Real &
machine_precision()
{
  static Real _machine_precision = 1E-15;
  return _machine_precision;
}

Real &
tolerance()
{
  static Real _tolerance = 1E-6;
  return _tolerance;
}

Real &
tighter_tolerance()
{
  static Real _tighter_tolerance = 1E-12;
  return _tighter_tolerance;
}

std::string &
buffer_name_separator()
{
  static std::string _buffer_sep = "_";
  return _buffer_sep;
}

std::string &
parameter_name_separator()
{
  static std::string _param_sep = "_";
  return _param_sep;
}
} // namespace neml2
