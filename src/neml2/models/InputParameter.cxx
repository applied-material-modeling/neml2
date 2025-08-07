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
#include "neml2/misc/assertions.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
template <typename T>
OptionSet
InputParameter<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "A parameter that is defined through an input variable. This object is not intended "
      "to be used directly in the input file.";

  options.set<bool>("define_second_derivatives") = true;

  options.set_input("from");
  options.set("from").doc() = "The input variable that defines this parameter";

  options.set_output("to");
  options.set("to").doc() = "The name of the parameter, default to 'parameters/object_name'";

  return options;
}

template <typename T>
InputParameter<T>::InputParameter(const OptionSet & options)
  : Model(options),
    _input_var(this->template declare_input_variable<T>("from")),
    _p(options.get("to").user_specified()
           ? this->template declare_output_variable<T>("to")
           : this->template declare_output_variable<T>(VariableName(PARAMETERS, this->name())))
{
}

template <typename T>
void
InputParameter<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    this->_p = _input_var();

  if (dout_din)
    if (_input_var.is_dependent())
      this->_p.d(_input_var) = imap<T>(_input_var.options()).template as<Tensor>();
}

#define REGISTER(T)                                                                                \
  using T##InputParameter = InputParameter<T>;                                                     \
  register_NEML2_object(T##InputParameter);                                                        \
  template class InputParameter<T>
FOR_ALL_PRIMITIVETENSOR(REGISTER);
} // namespace neml2
