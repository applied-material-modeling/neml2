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

#include "neml2/user_tensors/FullTensor.h"

namespace neml2
{
register_NEML2_object(FullTensor);

OptionSet
FullTensor::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() =
      "Construct a full Tensor with given batch and base shapes filled with a given value.";

  options.set<TensorShape>("batch_shape") = {};
  options.set("batch_shape").doc() = "Batch shape";

  options.set<TensorShape>("base_shape") = {};
  options.set("base_shape").doc() = "Base shape";

  options.set<double>("value");
  options.set("value").doc() = "Value used to fill the tensor";

  return options;
}

FullTensor::FullTensor(const OptionSet & options)
  : Tensor(Tensor::full(options.get<TensorShape>("batch_shape"),
                        options.get<TensorShape>("base_shape"),
                        options.get<double>("value"),
                        default_tensor_options())),
    UserTensorBase(options)
{
}
} // namespace neml2
