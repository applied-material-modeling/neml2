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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/models/ListUnsqueeze.h"

namespace neml2
{
template <typename T>
OptionSet
ListUnsqueeze<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = Model::expected_options();
  options.doc() = "List unsqueeze a " + tensor_type +
                  " by adding num single dimensions to the end of the current batch size";

  options.set<Size>("num") = 1;
  options.set("num").doc() = "Number of dimensions to unsqueeze at the end of the batch size";

  options.set_input("argument");
  options.set("argument").doc() = "Input tensor to expand";

  return options;
}

template <typename T>
ListUnsqueeze<T>::ListUnsqueeze(const OptionSet & options)
  : Model(options),
    _num(options.get<Size>("num")),
    _in(this->template declare_input_variable<T>("input")),
    _out(this->template declare_output_variable<T>(VariableName(PARAMETERS, name())))
{
}

template <typename T>
void
ListUnsqueeze<T>::set_value(bool out, bool dout_din, bool d1out_din2)
{
  auto size = _in.batch_sizes();
  for (auto i = 0; i < _num; i++)
    size.push_back(1);

  if (out)
    _out = _in.tensor().batch_expand(size);

  if (dout_din)
    _out.d(_in) = Tensor::identity(size, _in.batch_dim(), _in.options());

  if (d1out_din2)
  {
    // zero
  }
}

#define REGISTER(T)                                                                                \
  using T##ListUnsqueeze = ListUnsqueeze<T>;                                                       \
  register_NEML2_object(T##ListUnsqueeze);                                                         \
  template class ListUnsqueeze<T>
REGISTER(Scalar);
REGISTER(Vec);
REGISTER(SR2);
}