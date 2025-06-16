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

#include "neml2/user_tensors/LogspaceTensor.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(LogspaceTensor);

OptionSet
LogspaceTensor::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a Tensor with exponents linearly spaced on the batch dimensions. "
                  "See neml2::TensorBase::logspace for a detailed explanation.";

  options.set<TensorName<Tensor>>("start");
  options.set("start").doc() = "The starting tensor";

  options.set<TensorName<Tensor>>("end");
  options.set("end").doc() = "The ending tensor";

  options.set<Size>("nstep");
  options.set("nstep").doc() = "The number of steps with even spacing along the new dimension";

  options.set<Size>("dim") = 0;
  options.set("dim").doc() = "Where to insert the new dimension";

  options.set<double>("base") = 10;
  options.set("base").doc() = "Exponent base";

  options.set<Size>("batch_dim") = -1;
  options.set("batch_dim").doc() = "Batch dimension of the output";

  options.set<TensorShape>("batch_expand") = TensorShape();
  options.set("batch_expand").doc() = "After construction, perform an additional batch expanding "
                                      "operation into the given batch shape.";

  return options;
}

LogspaceTensor::LogspaceTensor(const OptionSet & options)
  : UserTensorBase(options),
    Tensor(make(options))
{
}

Tensor
LogspaceTensor::make(const OptionSet & options) const
{
  auto * f = this->factory();
  neml_assert(f, "Internal error: factory != nullptr");

  auto t = Tensor::logspace(options.get<TensorName<Tensor>>("start").resolve(f),
                            options.get<TensorName<Tensor>>("end").resolve(f),
                            options.get<Size>("nstep"),
                            options.get<Size>("dim"));

  // Change batch dimension if requested
  auto B = options.get<Size>("batch_dim");
  if (B >= 0)
    t = Tensor(t, B);

  // Expand if requested
  auto S = options.get<TensorShape>("batch_expand");
  if (!S.empty())
    t = t.batch_expand(S);

  return t;
}
} // namespace neml2
