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

#include "neml2/user_tensors/FromTorchScript.h"
#include "neml2/tensors/assertions.h"

#include <torch/script.h>
#include <torch/serialize.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace neml2
{
OptionSet
FromTorchScript::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Get the tensor from torch script. The torch scrip should have the "
                  "named_buffers and the associated tensor. Refer to "
                  "tests/regression/liquid_infiltration/gold/generate_load_file.py for an example";

  options.set<std::string>("pytorch_pt_file");
  options.set("pytorch_pt_file").doc() = "Name of the torch script file.";

  options.set<std::string>("tensor_name");
  options.set("tensor_name").doc() = "Associated named_buffers to extract the tensor from.";
  return options;
}

FromTorchScript::FromTorchScript(const OptionSet & options)
  : UserTensorBase(options)
{
}

torch::Tensor
FromTorchScript::load_torch_tensor(const OptionSet & options) const
{
  const auto fname = fs::path(options.get<std::string>("pytorch_pt_file"));
  const auto tensor_name = options.get<std::string>("tensor_name");
  const auto data = torch::jit::load(fname);

  torch::Tensor t;
  bool found = false;
  for (auto item : data.named_buffers())
  {
    if (item.name == tensor_name)
    {
      t = item.value;
      found = true;
      break;
    }
  }

  neml_assert(found, "No buffer named '", tensor_name, "' in file ", fname);
  t = t.to(torch::kFloat64);
  return t;
}
} // namespace neml2
