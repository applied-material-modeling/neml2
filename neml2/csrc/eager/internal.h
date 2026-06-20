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

// NON-shipped internal header: the opaque `Model::Impl` definition. Holds the
// pybind adapter object + cached metadata. Excluded from the wheel
// (wheel.exclude) and from the installed FILE_SET HEADERS, so the public ABI
// never sees pybind/Python types.

#include <string>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <pybind11/pybind11.h>

#include "neml2/csrc/eager/Model.h"
#include "neml2/csrc/eager/interpreter.h"

namespace neml2::eager
{
struct Model::Impl
{
  // Drops the pybind adapter reference under the GIL; defined in Model.cpp.
  ~Impl();

  // `interp` is declared FIRST so it is constructed first -- the embedded
  // interpreter must be running before `adapter` is imported/created.
  InterpreterGuard interp;

  // The `neml2.eager._EagerModel` instance that backs every operation. Reading
  // / destroying it requires the GIL.
  pybind11::object adapter;

  // Metadata cached once at load (under the GIL) so the accessors are GIL-free.
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<int> input_sizes;
  std::vector<int> output_sizes;
  at::Device device{at::kCPU};
  at::ScalarType dtype{at::kDouble};
};
} // namespace neml2::eager
