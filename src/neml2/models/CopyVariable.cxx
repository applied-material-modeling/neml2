// Copyright 2023, UChicago Argonne, LLC
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

namespace neml2
{
#define COPYVARIABLE_REGISTER_FIXEDDIMTENSOR(T) register_NEML2_object(Copy##T)
FOR_ALL_FIXEDDIMTENSOR(COPYVARIABLE_REGISTER_FIXEDDIMTENSOR);

template <typename T>
OptionSet
CopyVariable<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("from");
  options.set<VariableName>("to");
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

  if (dout_din)
    _to.d(_from) = T::identity_map(options());

  if (d2out_din2)
  {
    // zero
  }
}

#define COPYVARIABLE_INSTANTIATE_FIXEDDIMTENSOR(T) template class CopyVariable<T>
FOR_ALL_FIXEDDIMTENSOR(COPYVARIABLE_INSTANTIATE_FIXEDDIMTENSOR);
} // namespace neml2
