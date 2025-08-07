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

#include "neml2/models/DynamicSum.h"
#include "neml2/tensors/macros.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/sum.h"

namespace neml2
{
template <typename T>
OptionSet
DynamicSum<T>::expected_options()
{
  OptionSet options = Reduction<T>::expected_options();
  options.doc() = "Sum a dynamic dimension";
  return options;
}

template <typename T>
DynamicSum<T>::DynamicSum(const OptionSet & options)
  : Reduction<T>(options)
{
}

template <typename T>
void
DynamicSum<T>::set_value(bool out, bool /*dout_din*/, bool /*d2out_din2*/)
{
  if (out)
    _to = dynamic_sum(_from(), _dim);
}

#define REGISTER_DYNAMICSUM(T)                                                                     \
  using T##DynamicSum = DynamicSum<T>;                                                             \
  register_NEML2_object(T##DynamicSum);                                                            \
  template class DynamicSum<T>
FOR_ALL_PRIMITIVETENSOR(REGISTER_DYNAMICSUM);
} // namespace neml2
