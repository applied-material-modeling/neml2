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

#include "neml2/tensors/TensorValue.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <typename T>
void
TensorValue<T>::to_(const TensorOptions & options)
{
  _value = _value.to(options);
}

template <typename T>
void
TensorValue<T>::requires_grad_(bool req)
{
  _value.requires_grad_(req);
}

template <typename T>
TensorValue<T>::operator Tensor() const
{
  return _value;
}

template <typename T>
void
TensorValue<T>::operator=(const Tensor & val)
{
  _value = T(val);
}

template <typename T>
TensorType
TensorValue<T>::type() const
{
  return TensorTypeEnum<T>::value;
}

#define INSTANTIATE_TENSORVALUE(T) template class TensorValue<T>
FOR_ALL_TENSORBASE(INSTANTIATE_TENSORVALUE);
} // namespace neml2
