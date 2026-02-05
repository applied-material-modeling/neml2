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
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/logspace.h"

namespace neml2
{
template <typename T>
OptionSet
LogspaceTensorTmpl<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();
  options.doc() = "Construct a " + UserTensorBase<T>::tensor_type() +
                  " logarithmically spaced on the batch dimensions. See neml2::dynamic_logspace, "
                  "neml2::intmd_logspace, or neml2::base_logspace for a detailed explanation.";

  options.set<TensorName<T>>("start");
  options.set("start").doc() = "The starting tensor";

  options.set<TensorName<T>>("end");
  options.set("end").doc() = "The ending tensor";

  options.set<Size>("nstep");
  options.set("nstep").doc() = "The number of steps with even spacing along the new dimension";

  options.set<Size>("dim") = 0;
  options.set("dim").doc() = "Where to insert the new dimension";

  options.set<double>("base") = 10.0;
  options.set("base").doc() = "The base of the logarithm";

  EnumSelection selection({"dynamic", "intermediate"}, "dynamic");
  options.set<EnumSelection>("group") = selection;
  options.set("group").doc() =
      "Dimension group to apply the operation. Options are: " + selection.join();

  return options;
}

template <typename T>
LogspaceTensorTmpl<T>::LogspaceTensorTmpl(const OptionSet & options)
  : UserTensorBase<T>(options),
    _start(options.get<TensorName<T>>("start")),
    _end(options.get<TensorName<T>>("end")),
    _nstep(options.get<Size>("nstep")),
    _dim(options.get<Size>("dim")),
    _base(options.get<double>("base")),
    _group(options.get<EnumSelection>("group"))
{
}

template <typename T>
T
LogspaceTensorTmpl<T>::make() const
{
  auto * f = this->factory();
  neml_assert(f,
              "Internal error: factory is null while resolving tensor names. Ensure this user "
              "tensor is created via the NEML2 factory.");

  if (_group == "dynamic")
    return dynamic_logspace(_start.resolve(f), _end.resolve(f), _nstep, _dim, _base);
  else if (_group == "intermediate")
    return intmd_logspace(_start.resolve(f), _end.resolve(f), _nstep, _dim, _base);

  throw NEMLException("Internal error: invalid group selection");
}

#define LOGSPACETENSOR_REGISTER(T)                                                                 \
  using Logspace##T = LogspaceTensorTmpl<T>;                                                       \
  register_NEML2_object_alias(Logspace##T, "Logspace" #T)
FOR_ALL_TENSORBASE(LOGSPACETENSOR_REGISTER);
} // namespace neml2
