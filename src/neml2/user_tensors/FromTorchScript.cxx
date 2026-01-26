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

namespace neml2
{
#define REGISTER_FROMTORCHSCRIPT(T)                                                                \
  using T##FromTorchScript = FromTorchScript<T>;                                                   \
  register_NEML2_object(T##FromTorchScript)
FOR_ALL_PRIMITIVETENSOR(REGISTER_FROMTORCHSCRIPT);
#undef REGISTER_FROMTORCHSCRIPT

template <class T>
OptionSet
FromTorchScript<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();
  options.doc() = "Get the tensor from torch script. The torch script should define a Module with "
                  "named_buffers that stores the tensor to load. Refer to "
                  "tests/regression/liquid_infiltration/gold/generate_load_file.py for an example";

  options.set<std::string>("torch_script");
  options.set("torch_script").doc() = "Name of the torch script file.";

  options.set<std::string>("tensor_name");
  options.set("tensor_name").doc() = "Key of named_buffers to extract the tensor from.";

  options.set<TensorShape>("batch_shape") = {};
  options.set("batch_shape").doc() = "Batch shape";

  options.set<unsigned int>("intermediate_dimension") = 0;
  options.set("intermediate_dimension").doc() = "Intermediate dimension";

  options.set<TensorShape>("base_shape") = {};
  options.set("base_shape").doc() = "Base shape";

  if constexpr (!std::is_same_v<T, Tensor>)
  {
    options.set<TensorShape>("base_shape") = T::const_base_sizes;
    options.set("base_shape").suppressed() = true;
  }

  return options;
}

template <class T>
FromTorchScript<T>::FromTorchScript(const OptionSet & options)
  : UserTensorBase<T>(options),
    _torch_script(options.get<std::string>("torch_script")),
    _tensor_name(options.get<std::string>("tensor_name")),
    _batch_sizes(options.get<TensorShape>("batch_shape")),
    _base_sizes(options.get<TensorShape>("base_shape")),
    _intmd_dim(options.get<unsigned int>("intermediate_dimension"))
{
  neml_assert(_intmd_dim <= _batch_sizes.size(),
              "Intermediate dimension ",
              _intmd_dim,
              " must be less than or equal to the number of batch dimensions ",
              _batch_sizes.size());
}

template <class T>
T
FromTorchScript<T>::make() const
{
  const auto module = torch::jit::load(_torch_script);

  ATensor t;
  for (auto item : module.named_buffers())
  {
    if (item.name == _tensor_name)
    {
      t = item.value;
      break;
    }
  }

  if (!t.defined())
  {
    std::stringstream ss;
    for (auto item : module.named_buffers())
      ss << item.name << " ";
    throw NEMLException("No buffer named '" + _tensor_name +
                        "' in the module defined by torch script " + std::string(_torch_script) +
                        "\nAvailable buffers: " + ss.str());
  }

  t = t.to(default_tensor_options());

  auto B = TensorShapeRef(_batch_sizes);
  auto D = B.slice(0, B.size() - _intmd_dim);
  return T(t, D, _intmd_dim);
}
} // namespace neml2
