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

#include "neml2/models/InputParameter.h"

namespace neml2
{
#define INPUTPARAMETER_REGISTER(T) register_NEML2_object(T##InputParameter)
FOR_ALL_PRIMITIVETENSOR(INPUTPARAMETER_REGISTER);

template <typename T>
OptionSet
InputParameter<T>::expected_options()
{
  OptionSet options = NonlinearParameter<T>::expected_options();
  options.doc() = "A parameter that is defined through an input variable. This essentially "
                  "converts a nonlinear parameter to an input variable";
  options.set_input("from");
  options.set("from").doc() = "The input variable that defines this nonlinear parameter";
  return options;
}

template <typename T>
InputParameter<T>::InputParameter(const OptionSet & options)
  : NonlinearParameter<T>(options),
    _input_var(this->template declare_input_variable<T>("from"))
{
  neml_assert(utils::stringify(_input_var.name()) != this->name(),
              "InputParameter must use an input variable name different from the parameter name. "
              "They both has name '",
              this->name(),
              "'.");
}

template <typename T>
void
InputParameter<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    this->_p = _input_var.value();

  if (dout_din)
    if (_input_var.is_dependent())
      this->_p.d(_input_var) = T::identity_map(_input_var.options());

  // This is zero
  (void)d2out_din2;
}
} // namespace neml2
