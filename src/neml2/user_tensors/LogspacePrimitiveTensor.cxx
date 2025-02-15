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

#include "neml2/user_tensors/LogspacePrimitiveTensor.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <typename T>
OptionSet
LogspacePrimitiveTensor<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a " + tensor_type +
                  " with exponents linearly spaced on the batch dimensions. See "
                  "neml2::TensorBase::logspace for a detailed explanation.";

  options.set<TensorName>("start");
  options.set("start").doc() = "The starting tensor";

  options.set<TensorName>("end");
  options.set("end").doc() = "The ending tensor";

  options.set<Size>("nstep");
  options.set("nstep").doc() = "The number of steps with even spacing along the new dimension";

  options.set<Size>("dim") = 0;
  options.set("dim").doc() = "Where to insert the new dimension";

  options.set<Real>("base") = 10;
  options.set("base").doc() = "Exponent base";

  options.set<TensorShape>("batch_expand") = TensorShape();
  options.set("batch_expand").doc() = "After construction, perform an additional batch expanding "
                                      "operation into the given batch shape.";

  return options;
}

template <typename T>
LogspacePrimitiveTensor<T>::LogspacePrimitiveTensor(const OptionSet & options)
  : T(make(options)),
    UserTensorBase(options)
{
}

template <typename T>
T
LogspacePrimitiveTensor<T>::make(const OptionSet & options) const
{
  auto t = T::logspace(T(options.get<TensorName>("start")),
                       T(options.get<TensorName>("end")),
                       options.get<Size>("nstep"),
                       options.get<Size>("dim"),
                       options.get<Real>("base"));

  // Expand if requested
  auto S = options.get<TensorShape>("batch_expand");
  if (!S.empty())
    t = t.batch_expand(S);

  return t;
}

#define LOGSPACEPRIMITIVETENSOR_REGISTER(T)                                                        \
  using Logspace##T = LogspacePrimitiveTensor<T>;                                                  \
  register_NEML2_object_alias(Logspace##T, "Logspace" #T)
FOR_ALL_PRIMITIVETENSOR(LOGSPACEPRIMITIVETENSOR_REGISTER);
} // namespace neml2
