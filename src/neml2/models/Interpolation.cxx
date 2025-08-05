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

#include "neml2/models/Interpolation.h"
#include "neml2/misc/string_utils.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <typename T>
OptionSet
Interpolation<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = Model::expected_options();
  options.doc() = "Interpolate a " + tensor_type +
                  " as a function of the given argument. See neml2::Interpolation for rules on "
                  "shapes of the interpolant and the argument.";

  options.set<TensorName<T>>("ordinate");
  options.set("ordinate").doc() = tensor_type + " defining the ordinate values of the interpolant";

  options.set_output("output");
  options.set("output").doc() = tensor_type + " output of the interpolant";

  return options;
}

template <typename T>
Interpolation<T>::Interpolation(const OptionSet & options)
  : Model(options),
    _Y(this->template declare_parameter<T>("Y", "ordinate")),
    _p(options.get("output").user_specified()
           ? this->template declare_output_variable<T>("output")
           : this->template declare_output_variable<T>(VariableName(PARAMETERS, name())))
{
}

#define INSTANTIATE(T) template class Interpolation<T>
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE);
} // namespace neml2
