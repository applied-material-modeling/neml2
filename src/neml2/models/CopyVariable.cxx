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

#include "neml2/models/CopyVariable.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <typename T>
OptionSet
CopyVariable<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Copy the value from one variable to another.";

  options.set_input("from");
  options.set("from").doc() = "Variable to copy value from";

  options.set_output("to");
  options.set("to").doc() = "Variable to copy value to";

  return options;
}

template <typename T>
CopyVariable<T>::CopyVariable(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<T>("to")),
    _from(declare_input_variable<T>("from"))
{
}

template <typename T>
void
CopyVariable<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _to = T(_from);

  if (_from.is_dependent())
    if (dout_din)
      _to.d(_from) = T::identity_map(_from.options());

  if (d2out_din2)
  {
    // zero
  }
}

#define REGISTER_COPYVARIABLE(T)                                                                   \
  using Copy##T = CopyVariable<T>;                                                                 \
  register_NEML2_object(Copy##T);                                                                  \
  template class CopyVariable<T>
FOR_ALL_PRIMITIVETENSOR(REGISTER_COPYVARIABLE);
} // namespace neml2
